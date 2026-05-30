"""
ONNX Runtime inference service — loaded once at Django startup via AppConfig.
Includes occlusion-sensitivity heatmap generation to highlight affected regions.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

import onnxruntime as ort

# Correct Bethesda System terminology
CLASS_NAMES = {
    0: "Normal - NILM",
    1: "Atypical Squamous Cells (ASC-US)",
    2: "Low-Grade Squamous Intraepithelial Lesion (LSIL)",
    3: "High-Grade Squamous Intraepithelial Lesion (HSIL)",
    4: "Carcinoma"
}

# Clinical interpretation labels (NO "Stage" terminology - HSIL is precancer, not cancer stage)
STAGE_LABELS = {
    0: "Normal - Negative for Intraepithelial Lesion",
    1: "ASC-US - Atypical Cells Present",
    2: "LSIL - Low-Grade Precancerous Changes",
    3: "HSIL - High-Grade Precancerous Changes",
    4: "Carcinoma - Invasive Cancer Detected",
}

RISK_LEVELS = {
    0: {"level": "No Risk", "color": "#22c55e", "icon": "shield-check"},
    1: {"level": "Low Risk", "color": "#84cc16", "icon": "info"},
    2: {"level": "Moderate Risk", "color": "#eab308", "icon": "alert-triangle"},
    3: {"level": "High Risk", "color": "#f97316", "icon": "alert-circle"},
    4: {"level": "Critical Risk", "color": "#ef4444", "icon": "alert-octagon"},
}

RECOMMENDATIONS = {
    "Normal - NILM": "Routine screening according to guidelines (typically 3 years).",
    "Atypical Squamous Cells (ASC-US)": "HPV testing recommended. If HPV-positive, colposcopy evaluation.",
    "Low-Grade Squamous Intraepithelial Lesion (LSIL)": "Colposcopy recommended, especially if HPV-positive or persistent.",
    "High-Grade Squamous Intraepithelial Lesion (HSIL)": "Colposcopic evaluation with biopsy and treatment recommended.",
    "Carcinoma": "Immediate referral to gynecologic oncologist for staging and treatment planning.",
}

CLINICAL_EXPLANATIONS = {
    "Normal - NILM": (
        "No epithelial abnormality detected. The cells appear morphologically normal with "
        "no signs of dysplasia or malignancy. The nucleus-to-cytoplasm ratio is within normal "
        "limits, and cell boundaries are well-defined. This is a negative screening result."
    ),
    "Atypical Squamous Cells (ASC-US)": (
        "Atypical squamous cells are present, but the changes are unclear. This may represent "
        "benign reactive changes or early precancerous changes. HPV testing is typically recommended "
        "to determine if colposcopy is needed. Most ASC-US cases are benign, but follow-up is important."
    ),
    "Low-Grade Squamous Intraepithelial Lesion (LSIL)": (
        "Low-grade precancerous changes detected. LSIL represents mild dysplasia with early abnormal "
        "cell changes, often associated with HPV infection. Many LSIL cases resolve spontaneously "
        "without treatment. However, colposcopic evaluation is recommended to rule out higher-grade "
        "changes, especially if HPV-positive or persistent."
    ),
    "High-Grade Squamous Intraepithelial Lesion (HSIL)": (
        "High-grade precancerous changes detected. HSIL represents moderate to severe dysplasia with "
        "abnormal cells that have a higher risk of progressing to invasive cancer if left untreated. "
        "IMPORTANT: HSIL is precancerous and treatable — it is NOT invasive cancer. Colposcopic "
        "evaluation with biopsy and treatment is strongly recommended to prevent progression."
    ),
    "Carcinoma": (
        "Features consistent with invasive carcinoma identified. This indicates that abnormal cells "
        "may have invaded through the basement membrane into surrounding tissue. This is the most "
        "severe finding requiring immediate specialist referral for definitive diagnosis, staging, "
        "and treatment planning."
    ),
}

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceService:
    _instance: Optional["InferenceService"] = None

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.temperature: float = 1.0  # default (no scaling)

    @classmethod
    def get(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def load(self):
        from django.conf import settings
        model_path = Path(settings.MODEL_DIR) / "cervistage_net.onnx"
        temp_path = Path(settings.MODEL_DIR) / "temperature.json"
        print(f"[DEBUG] Looking for model at: {model_path}")
        print(f"[DEBUG] Model exists: {model_path.exists()}")

        if not model_path.exists():
            print(f"[ERROR] Model not found at {model_path}")
            return

        try:
            providers = ["CPUExecutionProvider"]  # Use CPU only
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"[SUCCESS] Model loaded from {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

        # Load temperature scaling parameter (produced by v2 training notebook)
        if temp_path.exists():
            try:
                with open(temp_path) as f:
                    self.temperature = float(json.load(f)["temperature"])
                print(f"[SUCCESS] Temperature scaling loaded: T={self.temperature:.4f}")
            except Exception as e:
                print(f"[WARN] Could not load temperature.json: {e}")
                self.temperature = 1.0
        else:
            print("[INFO] No temperature.json found — using T=1.0 (no calibration)")

    # ── preprocessing ──────────────────────────────────────────────────────────
    def preprocess(self, image_file) -> np.ndarray:
        img = Image.open(image_file).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    def _raw_image(self, image_file) -> np.ndarray:
        """Return the uploaded image as (224, 224, 3) uint8 RGB for heatmap overlay."""
        img = Image.open(image_file).convert("RGB").resize((224, 224), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    # ── confidence interpretation ─────────────────────────────────────────────────
    @staticmethod
    def _get_confidence_interpretation(confidence: float, uncertainty: float) -> Dict:
        """
        Interpret model confidence for clinical use.
        Returns interpretation level and recommended actions.

        Thresholds:
        - HIGH_CONFIDENCE: ≥80% confidence AND <30% uncertainty
        - MODERATE_CONFIDENCE: 60-79% confidence OR <50% uncertainty
        - LOW_CONFIDENCE: <60% confidence OR ≥50% uncertainty
        """
        if confidence >= 0.80 and uncertainty < 0.3:
            return {
                "level": "HIGH_CONFIDENCE",
                "message": "High Confidence Finding",
                "action": "Standard clinical protocol may be followed",
                "review_required": False,
                "color": "#10b981",  # Green
                "icon": "check-circle"
            }
        elif 0.60 <= confidence < 0.80 or uncertainty < 0.5:
            return {
                "level": "MODERATE_CONFIDENCE",
                "message": "Moderate Confidence - Review Recommended",
                "action": "Pathologist review recommended before clinical decisions",
                "review_required": True,
                "color": "#f59e0b",  # Amber/Orange
                "icon": "exclamation-triangle"
            }
        else:  # confidence < 0.60 or uncertainty >= 0.5
            return {
                "level": "LOW_CONFIDENCE",
                "message": "Low Confidence - Pathologist Review Required",
                "action": "MANDATORY: Pathologist review required before clinical decisions",
                "review_required": True,
                "color": "#ef4444",  # Red
                "icon": "alert-circle"
            }

    # ── core prediction ────────────────────────────────────────────────────────
    def predict(self, image_file) -> Dict:
        if self.session is None:
            raise RuntimeError(
                "Model not loaded. Copy cervistage_net.onnx to the models/ directory."
            )

        arr = self.preprocess(image_file)
        logits = self.session.run(None, {"input": arr})[0][0]

        # Apply temperature scaling for calibrated confidence
        scaled_logits = logits / self.temperature
        exp = np.exp(scaled_logits - scaled_logits.max())
        probs = exp / exp.sum()
        predicted_class = int(probs.argmax())
        label = CLASS_NAMES[predicted_class]

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        uncertainty = float(entropy / np.log(5))

        if uncertainty < 0.2:
            confidence_level = "High"
        elif uncertainty < 0.4:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        risk = RISK_LEVELS[predicted_class]

        # Get confidence-based interpretation
        confidence_interp = self._get_confidence_interpretation(
            round(float(probs.max()), 4),
            round(uncertainty, 4)
        )

        return {
            "predicted_class": predicted_class,
            "predicted_label": label,
            "stage_label": STAGE_LABELS[predicted_class],
            "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "confidence": round(float(probs.max()), 4),
            "uncertainty": round(uncertainty, 4),
            "confidence_level": confidence_level,
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "confidence_interpretation": confidence_interp,
            "recommendation": RECOMMENDATIONS.get(label, "Consult with a healthcare provider."),
            "explanation": CLINICAL_EXPLANATIONS.get(label, "Detailed clinical explanation not available."),
        }

    # ── occlusion-sensitivity heatmap ──────────────────────────────────────────
    def generate_heatmap(self, image_file, predicted_class: int,
                         grid_size: int = 8, patch_ratio: float = 0.15) -> str:
        """
        Compute an occlusion-sensitivity heatmap and overlay it on the original
        image.  Returns a base64-encoded PNG string.

        How it works:
          • Slide a grey square across the image in an NxN grid.
          • For each position, replace the patch with the mean pixel value and
            run inference.
          • Measure how much the predicted-class probability *drops*.
          • Regions where the drop is large are the most important — these are
            coloured red in the overlay.
        """
        if self.session is None:
            return ""

        try:
            image_file.seek(0)
            arr = self.preprocess(image_file)                   # (1,3,224,224)
            image_file.seek(0)
            raw_img = self._raw_image(image_file)               # (224,224,3) uint8

            # Baseline probability for the predicted class
            base_logits = self.session.run(None, {"input": arr})[0][0]
            base_exp = np.exp(base_logits - base_logits.max())
            base_probs = base_exp / base_exp.sum()
            base_prob = base_probs[predicted_class]

            patch_size = max(int(224 * patch_ratio), 16)
            stride = 224 // grid_size
            importance = np.zeros((grid_size, grid_size), dtype=np.float32)

            for gy in range(grid_size):
                for gx in range(grid_size):
                    y0 = gy * stride
                    x0 = gx * stride
                    y1 = min(y0 + patch_size, 224)
                    x1 = min(x0 + patch_size, 224)

                    occluded = arr.copy()
                    occluded[:, :, y0:y1, x0:x1] = 0.0  # grey (after normalisation ≈ mean)

                    occ_logits = self.session.run(None, {"input": occluded})[0][0]
                    occ_exp = np.exp(occ_logits - occ_logits.max())
                    occ_probs = occ_exp / occ_exp.sum()

                    # Importance = how much probability dropped
                    importance[gy, gx] = max(base_prob - occ_probs[predicted_class], 0.0)

            # Up-sample to 224×224 with smooth interpolation
            from PIL import ImageFilter as _IF
            imp_img = Image.fromarray(
                ((importance / (importance.max() + 1e-8)) * 255).astype(np.uint8)
            )
            imp_img = imp_img.resize((224, 224), Image.BILINEAR)
            imp_img = imp_img.filter(ImageFilter.GaussianBlur(radius=8))
            heatmap = np.array(imp_img, dtype=np.float32) / 255.0

            # Build colour overlay (JET-like: blue → green → yellow → red)
            overlay = self._apply_colormap(heatmap)  # (224,224,3)
            alpha = 0.45
            blended = (raw_img.astype(np.float32) * (1 - alpha) +
                       overlay.astype(np.float32) * alpha).astype(np.uint8)

            # Encode to base64 PNG
            pil_out = Image.fromarray(blended)
            buf = io.BytesIO()
            pil_out.save(buf, format="PNG")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return b64

        except Exception as e:
            print(f"[WARN] Heatmap generation failed: {e}")
            import traceback; traceback.print_exc()
            return ""

    @staticmethod
    def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
        """Pure-numpy JET-style colormap (no OpenCV dependency needed)."""
        h = heatmap[..., np.newaxis]          # (H, W, 1)
        # Piecewise linear JET approximation
        r = np.clip(1.5 - np.abs(h * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(h * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(h * 4 - 1), 0, 1)
        rgb = np.concatenate([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    # ── batch averaging ────────────────────────────────────────────────────────
    def average_predictions(self, predictions_list: list) -> Dict:
        """
        Average predictions from multiple images and return aggregated result.
        
        Args:
            predictions_list: List of prediction dicts from self.predict()
        
        Returns:
            Averaged prediction dict with final class, confidence, etc.
        """
        if not predictions_list:
            raise ValueError("At least one prediction is required")

        # Average all probabilities across all images
        num_images = len(predictions_list)
        avg_probs = np.zeros(5, dtype=np.float32)  # 5 classes
        
        print(f"[DEBUG] Averaging {num_images} predictions")
        
        for pred_idx, pred in enumerate(predictions_list):
            probs_dict = pred["probabilities"]
            print(f"[DEBUG] Prediction {pred_idx+1}: {probs_dict}")
            
            # Iterate through class indices 0-4 and get corresponding names
            for class_idx in range(5):
                class_name = CLASS_NAMES[class_idx]
                # Get probability for this class from the probabilities dict
                prob_value = probs_dict.get(class_name, 0.0)
                avg_probs[class_idx] += prob_value
        
        avg_probs = avg_probs / num_images
        
        print(f"[DEBUG] Averaged probabilities: {avg_probs}")
        print(f"[DEBUG] Averaged probs dict: {dict(zip(range(5), avg_probs))}")
        
        # Determine final predicted class from averaged probabilities
        predicted_class = int(avg_probs.argmax())
        label = CLASS_NAMES[predicted_class]
        
        # Calculate averaged confidence and uncertainty
        avg_confidence = float(avg_probs.max())
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
        avg_uncertainty = float(entropy / np.log(5))
        
        # Determine confidence level
        if avg_uncertainty < 0.2:
            confidence_level = "High"
        elif avg_uncertainty < 0.4:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        # Get risk level and other metadata
        risk = RISK_LEVELS[predicted_class]
        confidence_interp = self._get_confidence_interpretation(avg_confidence, avg_uncertainty)
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": label,
            "stage_label": STAGE_LABELS[predicted_class],
            "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(avg_probs)},
            "confidence": round(avg_confidence, 4),
            "uncertainty": round(avg_uncertainty, 4),
            "confidence_level": confidence_level,
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "confidence_interpretation": confidence_interp,
            "recommendation": RECOMMENDATIONS.get(label, "Consult with a healthcare provider."),
            "explanation": CLINICAL_EXPLANATIONS.get(label, "Detailed clinical explanation not available."),
        }
