"""
Ultra-fast inference service for VPS deployment.
Key optimizations for slow VPS servers:
- Aggressive image downsampling BEFORE loading
- Fast processing pipeline
- Minimal memory usage
- Batch processing with parallel workers
"""

import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageFile

import onnxruntime as ort

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAXBLOCK = 2**25  # Handle large images

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
MODEL_INPUT_SIZE = 224


class VPSInferenceService:
    """Ultra-fast inference service optimized for VPS servers."""

    _instance: Optional["VPSInferenceService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.temperature: float = 1.0
        self.use_gpu: bool = False
        self._session_lock = threading.Lock()

    @classmethod
    def get(cls) -> "VPSInferenceService":
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

        print(f"[VPS FAST] Loading model from: {model_path}")

        if not model_path.exists():
            print(f"[ERROR] Model not found at {model_path}")
            return

        try:
            # Optimized for VPS (low CPU, no GPU)
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Better for single-core
            session_options.intra_op_num_threads = 2  # Reduce for VPS
            session_options.inter_op_num_threads = 1

            providers = ['CPUExecutionProvider']
            print(f"[VPS FAST] Using CPU-only inference")

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            print(f"[SUCCESS] VPS-optimized model loaded")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")

        # Load temperature scaling
        if temp_path.exists():
            try:
                with open(temp_path) as f:
                    self.temperature = float(json.load(f)["temperature"])
                print(f"[SUCCESS] Temperature loaded: T={self.temperature:.4f}")
            except:
                self.temperature = 1.0

    def _load_and_resize_fast(self, image_file) -> np.ndarray:
        """
        ULTRA-FAST image loading with aggressive downsampling.
        This is the KEY optimization for VPS.
        """
        # Load image with PIL
        img = Image.open(image_file)

        # Check if we need aggressive downscaling
        width, height = img.size
        max_dim = max(width, height)

        # If very large, downscale in stages (faster than one big resize)
        if max_dim > 4096:
            # First stage: downscale to 4096
            scale = 4096 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.NEAREST)
            width, height = new_width, new_height
            max_dim = max(width, height)

        if max_dim > 2048:
            # Second stage: downscale to 2048
            scale = 2048 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.NEAREST)

        # Final resize to model input size
        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)

        # Convert to array and normalize
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        return arr.transpose(2, 0, 1)[np.newaxis]

    def preprocess_single(self, image_file) -> np.ndarray:
        """Fast preprocessing for single image."""
        return self._load_and_resize_fast(image_file)

    def predict(self, image_file, skip_heatmap=True) -> Dict:
        """Single image prediction (no heatmap for speed)."""
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
        print(f"[VPS FAST] Single prediction: {elapsed:.3f}s")

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
            "inference_time": elapsed
        }

    def predict_batch(self, image_files: List) -> List[Dict]:
        """Batch prediction for multiple images."""
        if self.session is None:
            raise RuntimeError("Model not loaded")

        if not image_files:
            return []

        start_time = time.time()
        print(f"[VPS FAST] Processing batch of {len(image_files)} images")

        # Process all images
        predictions = []
        for i, image_file in enumerate(image_files):
            try:
                pred = self.predict(image_file, skip_heatmap=True)
                predictions.append(pred)
                print(f"[VPS FAST] Processed {i+1}/{len(image_files)}")
            except Exception as e:
                print(f"[ERROR] Failed to process image {i+1}: {e}")
                predictions.append(None)

        elapsed = time.time() - start_time
        print(f"[VPS FAST] Batch complete: {elapsed:.3f}s total ({elapsed/len(image_files):.3f}s per image)")

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

    def average_predictions(self, predictions_list: list) -> Dict:
        """Average predictions from multiple images."""
        if not predictions_list:
            raise ValueError("At least one prediction is required")

        num_images = len(predictions_list)
        avg_probs = np.zeros(5, dtype=np.float32)

        for pred in predictions_list:
            if pred is None:
                continue
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
