"""
Fast ONNX Runtime inference service optimized for large images.
Key optimizations:
- Aggressive image resizing before processing
- Fast bilinear interpolation (instead of slow LANCZOS)
- Progressive JPEG handling
- Memory-efficient preprocessing
- Skip heatmap by default for batches
"""

import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile

import onnxruntime as ort

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Correct Bethesda System terminology
CLASS_NAMES = {
    0: "Normal - NILM",
    1: "Atypical Squamous Cells (ASC-US)",
    2: "Low-Grade Squamous Intraepithelial Lesion (LSIL)",
    3: "High-Grade Squamous Intraepithelial Lesion (HSIL)",
    4: "Carcinoma"
}

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
        "no signs of dysplasia or malignancy."
    ),
    "Atypical Squamous Cells (ASC-US)": (
        "Atypical squamous cells are present, but the changes are unclear. This may represent "
        "benign reactive changes or early precancerous changes."
    ),
    "Low-Grade Squamous Intraepithelial Lesion (LSIL)": (
        "Low-grade precancerous changes detected. LSIL represents mild dysplasia with early abnormal "
        "cell changes, often associated with HPV infection."
    ),
    "High-Grade Squamous Intraepithelial Lesion (HSIL)": (
        "High-grade precancerous changes detected. HSIL represents moderate to severe dysplasia. "
        "IMPORTANT: HSIL is precancerous and treatable — it is NOT invasive cancer."
    ),
    "Carcinoma": (
        "Features consistent with invasive carcinoma identified. This indicates that abnormal cells "
        "may have invaded through the basement membrane."
    ),
}

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Target size for model input (don't change this)
MODEL_INPUT_SIZE = 224

# Maximum size to read from uploaded files (larger images will be resized first)
MAX_READ_SIZE = 2048  # First resize to this max dimension


class FastInferenceService:
    """High-performance inference service optimized for large images."""

    _instance: Optional["FastInferenceService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.temperature: float = 1.0
        self.use_gpu: bool = False
        self.batch_size: int = 8
        self._session_lock = threading.Lock()

    @classmethod
    def get(cls) -> "FastInferenceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.load()
        return cls._instance

    def load(self):
        from django.conf import settings

        model_path = Path(settings.MODEL_DIR) / "cervistage_net.onnx"
        temp_path = Path(settings.MODEL_DIR) / "temperature.json"

        print(f"[FAST] Looking for model at: {model_path}")

        if not model_path.exists():
            print(f"[ERROR] Model not found at {model_path}")
            return

        try:
            # Configure ONNX Runtime for optimal performance
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 2

            # Try GPU first, fallback to CPU
            providers = []
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.use_gpu = True
                    print("[FAST] Using GPU acceleration (CUDA)")
                else:
                    providers = ['CPUExecutionProvider']
                    print("[FAST] Using CPU (no GPU available)")
            except:
                providers = ['CPUExecutionProvider']

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            print(f"[SUCCESS] Fast model loaded")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

        # Load temperature scaling
        if temp_path.exists():
            try:
                with open(temp_path) as f:
                    self.temperature = float(json.load(f)["temperature"])
                print(f"[SUCCESS] Temperature scaling loaded: T={self.temperature:.4f}")
            except Exception as e:
                print(f"[WARN] Could not load temperature.json: {e}")
                self.temperature = 1.0

    def _load_and_resize_image(self, image_file) -> Image.Image:
        """
        Load image with aggressive resizing for large files.
        This is the KEY optimization for large image handling.
        """
        # First, get image dimensions without loading full data
        img = Image.open(image_file)

        # Check if we need aggressive downscaling
        width, height = img.size
        max_dim = max(width, height)

        # If image is very large, do a first pass resize to reduce memory usage
        if max_dim > MAX_READ_SIZE:
            # Calculate scale factor
            scale = MAX_READ_SIZE / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Use fast NEAREST for first downscale (much faster than LANCZOS)
            img = img.resize((new_width, new_height), Image.NEAREST)
            print(f"[FAST] Downscaled {width}x{height} → {new_width}x{new_height}")

        # Now resize to model input size
        # Use BILINEAR instead of LANCZOS for speed (minimal quality difference)
        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)

        return img

    def preprocess_single(self, image_file) -> np.ndarray:
        """Fast preprocessing for single image."""
        img = self._load_and_resize_image(image_file)
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        return arr.transpose(2, 0, 1)[np.newaxis]

    def preprocess_batch(self, image_files) -> np.ndarray:
        """Fast batch preprocessing."""
        batch = []
        for img_file in image_files:
            img = self._load_and_resize_image(img_file)
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - IMAGE_MEAN) / IMAGE_STD
            batch.append(arr.transpose(2, 0, 1))
        return np.stack(batch, axis=0)

    def predict(self, image_file, skip_heatmap=True) -> Dict:
        """Single image prediction with heatmap option."""
        if self.session is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        arr = self.preprocess_single(image_file)

        with self._session_lock:
            logits = self.session.run(None, {"input": arr})[0][0]

        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        exp = np.exp(scaled_logits - scaled_logits.max())
        probs = exp / exp.sum()
        predicted_class = int(probs.argmax())
        label = CLASS_NAMES[predicted_class]

        # Calculate uncertainty
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        uncertainty = float(entropy / np.log(5))

        # Determine confidence level
        if uncertainty < 0.2:
            confidence_level = "High"
        elif uncertainty < 0.4:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        risk = RISK_LEVELS[predicted_class]
        confidence_interp = self._get_confidence_interpretation(
            round(float(probs.max()), 4),
            round(uncertainty, 4)
        )

        elapsed = time.time() - start_time
        print(f"[FAST] Single prediction: {elapsed:.3f}s")

        result = {
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
            "inference_time": elapsed
        }

        # Generate heatmap only if requested (slow operation)
        if not skip_heatmap:
            result["heatmap"] = self.generate_heatmap_fast(image_file, predicted_class)

        return result

    def predict_batch(self, image_files: List, skip_heatmap: bool = True) -> List[Dict]:
        """Batch prediction - MUCH FASTER than sequential."""
        if self.session is None:
            raise RuntimeError("Model not loaded")

        if not image_files:
            return []

        start_time = time.time()

        # Preprocess all images
        batch = self.preprocess_batch(image_files)
        batch_size = batch.shape[0]

        # Run inference on entire batch
        with self._session_lock:
            logits_batch = self.session.run(None, {"input": batch})[0]

        # Process results
        predictions = []
        for i in range(batch_size):
            logits = logits_batch[i]
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
            confidence_interp = self._get_confidence_interpretation(
                round(float(probs.max()), 4),
                round(uncertainty, 4)
            )

            predictions.append({
                "predicted_class": predicted_class,
                "predicted_label": label,
                "stage_label": STAGE_LABELS[predicted_class],
                "probabilities": {CLASS_NAMES[j]: round(float(p), 4) for j, p in enumerate(probs)},
                "confidence": round(float(probs.max()), 4),
                "uncertainty": round(uncertainty, 4),
                "confidence_level": confidence_level,
                "risk_level": risk["level"],
                "risk_color": risk["color"],
                "confidence_interpretation": confidence_interp,
                "recommendation": RECOMMENDATIONS.get(label, "Consult with a healthcare provider."),
                "explanation": CLINICAL_EXPLANATIONS.get(label, "Detailed clinical explanation not available."),
            })

        elapsed = time.time() - start_time
        print(f"[FAST] Batch prediction ({batch_size} images): {elapsed:.3f}s ({elapsed/batch_size:.3f}s per image)")

        return predictions

    def _get_confidence_interpretation(self, confidence: float, uncertainty: float) -> Dict:
        """Interpret model confidence for clinical use."""
        if confidence >= 0.80 and uncertainty < 0.3:
            return {
                "level": "HIGH_CONFIDENCE",
                "message": "High Confidence Finding",
                "action": "Standard clinical protocol may be followed",
                "review_required": False,
                "color": "#10b981",
                "icon": "check-circle"
            }
        elif 0.60 <= confidence < 0.80 or uncertainty < 0.5:
            return {
                "level": "MODERATE_CONFIDENCE",
                "message": "Moderate Confidence - Review Recommended",
                "action": "Pathologist review recommended before clinical decisions",
                "review_required": True,
                "color": "#f59e0b",
                "icon": "exclamation-triangle"
            }
        else:
            return {
                "level": "LOW_CONFIDENCE",
                "message": "Low Confidence - Pathologist Review Required",
                "action": "MANDATORY: Pathologist review required before clinical decisions",
                "review_required": True,
                "color": "#ef4444",
                "icon": "alert-circle"
            }

    def generate_heatmap_fast(self, image_file, predicted_class: int,
                              grid_size: int = 3) -> str:
        """Fast heatmap generation with low resolution."""
        if self.session is None:
            return ""

        try:
            image_file.seek(0)
            arr = self.preprocess_single(image_file)
            image_file.seek(0)
            raw_img = self._raw_image(image_file)

            # Baseline probability
            base_logits = self.session.run(None, {"input": arr})[0][0]
            base_exp = np.exp(base_logits - base_logits.max())
            base_probs = base_exp / base_exp.sum()
            base_prob = base_probs[predicted_class]

            patch_size = max(int(224 * 0.2), 16)
            stride = 224 // grid_size
            importance = np.zeros((grid_size, grid_size), dtype=np.float32)

            for gy in range(grid_size):
                for gx in range(grid_size):
                    y0 = gy * stride
                    x0 = gx * stride
                    y1 = min(y0 + patch_size, 224)
                    x1 = min(x0 + patch_size, 224)

                    occluded = arr.copy()
                    occluded[:, :, y0:y1, x0:x1] = 0.0

                    occ_logits = self.session.run(None, {"input": occluded})[0][0]
                    occ_exp = np.exp(occ_logits - occ_logits.max())
                    occ_probs = occ_exp / occ_probs.sum()

                    importance[gy, gx] = max(base_prob - occ_probs[predicted_class], 0.0)

            # Up-sample with smooth interpolation
            imp_img = Image.fromarray(
                ((importance / (importance.max() + 1e-8)) * 255).astype(np.uint8)
            )
            imp_img = imp_img.resize((224, 224), Image.BILINEAR)

            from PIL import ImageFilter
            imp_img = imp_img.filter(ImageFilter.GaussianBlur(radius=8))
            heatmap = np.array(imp_img, dtype=np.float32) / 255.0

            # Build colour overlay
            overlay = self._apply_colormap(heatmap)
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
            return ""

    @staticmethod
    def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
        """Pure-numpy JET-style colormap."""
        h = heatmap[..., np.newaxis]
        r = np.clip(1.5 - np.abs(h * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(h * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(h * 4 - 1), 0, 1)
        rgb = np.concatenate([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    def _raw_image(self, image_file) -> np.ndarray:
        """Return the uploaded image as (224, 224, 3) uint8 RGB."""
        img = self._load_and_resize_image(image_file)
        img = img.convert("RGB")
        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def average_predictions(self, predictions_list: list) -> Dict:
        """Average predictions from multiple images."""
        if not predictions_list:
            raise ValueError("At least one prediction is required")

        num_images = len(predictions_list)
        avg_probs = np.zeros(5, dtype=np.float32)

        for pred in predictions_list:
            probs_dict = pred["probabilities"]
            for class_idx in range(5):
                class_name = CLASS_NAMES[class_idx]
                prob_value = probs_dict.get(class_name, 0.0)
                avg_probs[class_idx] += prob_value

        avg_probs = avg_probs / num_images

        predicted_class = int(avg_probs.argmax())
        label = CLASS_NAMES[predicted_class]

        avg_confidence = float(avg_probs.max())
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
        avg_uncertainty = float(entropy / np.log(5))

        if avg_uncertainty < 0.2:
            confidence_level = "High"
        elif avg_uncertainty < 0.4:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

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
