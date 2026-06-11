"""
Unified ONNX Runtime inference service for CerviStage AI.
Combines all optimization strategies into a single configurable service.

Performance Modes:
- 'standard': Full-featured with heatmap generation (default)
- 'fast': Optimized for speed with aggressive downsampling
- 'ultra': Memory-efficient for large files on VPS/slow servers
"""

import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
from PIL import Image, ImageFilter, ImageFile

import onnxruntime as ort

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAXBLOCK = 2**25  # Handle large images

# ============================================================================
# CLINICAL CLASSIFICATIONS AND LABELS
# ============================================================================

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

# Image normalization constants (ImageNet pretrained model)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size (EfficientNet-B3 uses 224x224)
MODEL_INPUT_SIZE = 224

# ============================================================================
# PERFORMANCE MODE CONFIGURATIONS
# ============================================================================

class PerformanceMode:
    """Configuration for different performance modes."""

    # Standard mode: Full quality with heatmaps
    STANDARD = {
        'name': 'standard',
        'thumbnail_size': 1000,
        'resample': Image.LANCZOS,
        'execution_mode': ort.ExecutionMode.ORT_SEQUENTIAL,
        'intra_op_threads': 1,
        'inter_op_threads': 1,
        'heatmap_default': False,
    }

    # Fast mode: Optimized for speed with batch processing
    FAST = {
        'name': 'fast',
        'thumbnail_size': 2048,
        'resample': Image.BILINEAR,
        'execution_mode': ort.ExecutionMode.ORT_PARALLEL,
        'intra_op_threads': 4,
        'inter_op_threads': 2,
        'heatmap_default': True,
    }

    # Ultra mode: Memory-efficient for VPS/large files
    ULTRA = {
        'name': 'ultra',
        'thumbnail_size': 512,
        'resample': Image.BILINEAR,
        'execution_mode': ort.ExecutionMode.ORT_SEQUENTIAL,
        'intra_op_threads': 2,
        'inter_op_threads': 1,
        'heatmap_default': True,
    }


# ============================================================================
# UNIFIED INFERENCE SERVICE
# ============================================================================

class InferenceService:
    """
    Unified inference service with configurable performance modes.

    Usage:
        service = InferenceService.get()

        # Standard mode with heatmap
        result = service.predict(image_file, mode='standard', skip_heatmap=False)

        # Fast mode without heatmap
        result = service.predict(image_file, mode='fast', skip_heatmap=True)

        # Ultra mode for large files
        result = service.predict(image_file, mode='ultra')
    """

    _instance: Optional["InferenceService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.temperature: float = 1.0
        self._session_lock = threading.Lock()
        self._loaded_mode: Optional[str] = None

    @classmethod
    def get(cls) -> "InferenceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.load()
        return cls._instance

    def load(self, mode: Literal['standard', 'fast', 'ultra'] = 'standard'):
        """
        Load the ONNX model with specified performance mode.

        Args:
            mode: Performance mode - 'standard', 'fast', or 'ultra'
        """
        from django.conf import settings

        model_path = Path(settings.MODEL_DIR) / "cervistage_net.onnx"
        temp_path = Path(settings.MODEL_DIR) / "temperature.json"

        config = getattr(PerformanceMode, mode.upper(), PerformanceMode.STANDARD)

        print(f"[INFERENCE] Loading model in {mode} mode...")
        print(f"[INFERENCE] Model path: {model_path}")
        print(f"[INFERENCE] Model exists: {model_path.exists()}")

        if not model_path.exists():
            print(f"[ERROR] Model not found at {model_path}")
            return

        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = config['execution_mode']
            session_options.intra_op_num_threads = config['intra_op_threads']
            session_options.inter_op_num_threads = config['inter_op_threads']

            # Try GPU first, fallback to CPU
            providers = ['CPUExecutionProvider']
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print(f"[INFERENCE] GPU acceleration available (CUDA)")
            except:
                pass

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            self._loaded_mode = mode

            print(f"[SUCCESS] Model loaded in {mode} mode")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

        # Load temperature scaling parameter
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

    # ── Image Loading Strategies ─────────────────────────────────────────────────

    def _load_image_standard(self, image_file) -> np.ndarray:
        """Load image with high-quality LANCZOS resampling (standard mode)."""
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")

        # For very large images, create thumbnail first
        if img.size[0] > 1000 or img.size[1] > 1000:
            img.thumbnail((1000, 1000), Image.LANCZOS)

        # Resize to model input size
        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

        # Convert to numpy array and normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD

        return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    def _load_image_fast(self, image_file) -> np.ndarray:
        """
        Load image with aggressive downsampling (fast mode).
        Uses NEAREST for first pass, BILINEAR for final resize.
        """
        img = Image.open(image_file)
        width, height = img.size
        max_dim = max(width, height)

        # If image is very large, do aggressive first-pass resize
        if max_dim > 2048:
            scale = 2048 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.NEAREST)

        # Final resize to model input size
        img = img.convert("RGB")
        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        return arr.transpose(2, 0, 1)[np.newaxis]

    def _load_image_ultra(self, image_file) -> np.ndarray:
        """
        Ultra-fast loading using PIL thumbnail (ultra mode).
        Resizes DURING load for maximum memory efficiency.
        """
        image_file.seek(0)
        img = Image.open(image_file)

        # Use PIL's thumbnail which only loads necessary data
        img.thumbnail((512, 512), Image.BILINEAR)

        # Final resize
        if img.size[0] != MODEL_INPUT_SIZE or img.size[1] != MODEL_INPUT_SIZE:
            img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)

        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        return arr.transpose(2, 0, 1)[np.newaxis]

    def _load_image(self, image_file, mode: str) -> np.ndarray:
        """Load image using strategy based on performance mode."""
        if mode == 'standard':
            return self._load_image_standard(image_file)
        elif mode == 'fast':
            return self._load_image_fast(image_file)
        else:  # ultra
            return self._load_image_ultra(image_file)

    def _raw_image(self, image_file, mode: str) -> np.ndarray:
        """Return image as (224, 224, 3) uint8 RGB for heatmap overlay."""
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")

        config = getattr(PerformanceMode, mode.upper(), PerformanceMode.STANDARD)
        thumbnail_size = config['thumbnail_size']

        if img.size[0] > thumbnail_size or img.size[1] > thumbnail_size:
            img.thumbnail((thumbnail_size, thumbnail_size), Image.LANCZOS)

        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)

    # ── Confidence Interpretation ─────────────────────────────────────────────────

    @staticmethod
    def _get_confidence_interpretation(confidence: float, uncertainty: float) -> Dict:
        """
        Interpret model confidence for clinical use.

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

    # ── Core Prediction Methods ────────────────────────────────────────────────────

    def predict(
        self,
        image_file,
        mode: Literal['standard', 'fast', 'ultra'] = 'standard',
        skip_heatmap: Optional[bool] = None
    ) -> Dict:
        """
        Run prediction on a single image.

        Args:
            image_file: Uploaded image file
            mode: Performance mode ('standard', 'fast', 'ultra')
            skip_heatmap: Skip heatmap generation (None uses mode default)

        Returns:
            Prediction dict with class, probabilities, confidence, etc.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Copy cervistage_net.onnx to the models/ directory.")

        # Use mode's default for skip_heatmap if not specified
        if skip_heatmap is None:
            config = getattr(PerformanceMode, mode.upper(), PerformanceMode.STANDARD)
            skip_heatmap = config['heatmap_default']

        start_time = time.time()

        # Load and preprocess image
        arr = self._load_image(image_file, mode)

        # Run inference
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
        print(f"[INFERENCE] Prediction ({mode} mode): {elapsed:.3f}s")

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
        }

        # Add inference time for fast/ultra modes
        if mode != 'standard':
            result["inference_time"] = round(elapsed, 3)

        # Generate heatmap if requested
        if not skip_heatmap:
            result["heatmap"] = self.generate_heatmap(image_file, predicted_class, mode)

        return result

    def predict_batch(
        self,
        image_files: List,
        mode: Literal['standard', 'fast', 'ultra'] = 'fast',
        skip_heatmap: bool = True
    ) -> List[Dict]:
        """
        Run prediction on multiple images (optimized for batch processing).

        Args:
            image_files: List of uploaded image files
            mode: Performance mode ('fast' recommended for batches)
            skip_heatmap: Skip heatmap generation (recommended for batches)

        Returns:
            List of prediction dicts
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")

        if not image_files:
            return []

        start_time = time.time()
        predictions = []

        # For ultra mode, process sequentially (memory-efficient)
        if mode == 'ultra':
            for i, image_file in enumerate(image_files):
                try:
                    pred = self.predict(image_file, mode=mode, skip_heatmap=skip_heatmap)
                    predictions.append(pred)
                except Exception as e:
                    print(f"[ERROR] Failed to process image {i+1}: {e}")
                    predictions.append(None)
        else:
            # For standard/fast modes, can process in batches
            for image_file in image_files:
                try:
                    pred = self.predict(image_file, mode=mode, skip_heatmap=skip_heatmap)
                    predictions.append(pred)
                except Exception as e:
                    print(f"[ERROR] Failed to process image: {e}")
                    predictions.append(None)

        elapsed = time.time() - start_time
        print(f"[INFERENCE] Batch prediction ({mode} mode): {elapsed:.3f}s for {len(image_files)} images")

        return predictions

    # ── Heatmap Generation ───────────────────────────────────────────────────────

    def generate_heatmap(
        self,
        image_file,
        predicted_class: int,
        mode: Literal['standard', 'fast', 'ultra'] = 'standard',
        grid_size: int = 4,
        patch_ratio: float = 0.15
    ) -> str:
        """
        Generate occlusion-sensitivity heatmap overlay.

        Returns:
            Base64-encoded PNG string of the heatmap overlay.
        """
        if self.session is None:
            return ""

        try:
            image_file.seek(0)
            arr = self._load_image(image_file, mode)
            image_file.seek(0)
            raw_img = self._raw_image(image_file, mode)

            # Baseline probability for predicted class
            base_logits = self.session.run(None, {"input": arr})[0][0]
            base_exp = np.exp(base_logits - base_logits.max())
            base_probs = base_exp / base_exp.sum()
            base_prob = base_probs[predicted_class]

            patch_size = max(int(MODEL_INPUT_SIZE * patch_ratio), 16)
            stride = MODEL_INPUT_SIZE // grid_size
            importance = np.zeros((grid_size, grid_size), dtype=np.float32)

            for gy in range(grid_size):
                for gx in range(grid_size):
                    y0 = gy * stride
                    x0 = gx * stride
                    y1 = min(y0 + patch_size, MODEL_INPUT_SIZE)
                    x1 = min(x0 + patch_size, MODEL_INPUT_SIZE)

                    occluded = arr.copy()
                    occluded[:, :, y0:y1, x0:x1] = 0.0

                    occ_logits = self.session.run(None, {"input": occluded})[0][0]
                    occ_exp = np.exp(occ_logits - occ_logits.max())
                    occ_probs = occ_exp / occ_exp.sum()

                    importance[gy, gx] = max(base_prob - occ_probs[predicted_class], 0.0)

            # Up-sample and smooth
            imp_img = Image.fromarray(
                ((importance / (importance.max() + 1e-8)) * 255).astype(np.uint8)
            )
            imp_img = imp_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
            imp_img = imp_img.filter(ImageFilter.GaussianBlur(radius=8))
            heatmap = np.array(imp_img, dtype=np.float32) / 255.0

            # Apply colormap
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
        """Pure-numpy JET-style colormap (no OpenCV dependency)."""
        h = heatmap[..., np.newaxis]
        r = np.clip(1.5 - np.abs(h * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(h * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(h * 4 - 1), 0, 1)
        rgb = np.concatenate([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    # ── Prediction Averaging ───────────────────────────────────────────────────────

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

        num_images = len(predictions_list)
        avg_probs = np.zeros(5, dtype=np.float32)

        print(f"[INFERENCE] Averaging {num_images} predictions")

        for pred_idx, pred in enumerate(predictions_list):
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


# ============================================================================
# CONVENIENCE ALIASES FOR BACKWARD COMPATIBILITY
# ============================================================================

# These aliases allow existing code to work without changes
# They will be deprecated in future versions

def get_inference_service() -> InferenceService:
    """Get the singleton inference service instance."""
    return InferenceService.get()

# Backward compatibility: Old class names that now point to the same service
FastInferenceService = InferenceService  # Alias for backward compatibility
UltraFastInferenceService = InferenceService  # Alias for backward compatibility
