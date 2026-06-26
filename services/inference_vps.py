"""
Optimised inference service for VPS/server deployment.
Key speedups vs original:
- True ONNX batch inference (N images → one session.run call)
- Parallel image preprocessing via ThreadPoolExecutor
- Removed unnecessary session lock (ONNX Runtime is thread-safe)
- Model warm-up on load (avoids slow first request)
- Input name cached at load time
- Thread count tuned to available cores
"""

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageFile

import onnxruntime as ort

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    0: {"level": "No Risk",       "color": "#22c55e"},
    1: {"level": "Low Risk",      "color": "#84cc16"},
    2: {"level": "Moderate Risk", "color": "#eab308"},
    3: {"level": "High Risk",     "color": "#f97316"},
    4: {"level": "Critical Risk", "color": "#ef4444"},
}

RECOMMENDATIONS = {
    "Normal - NILM": "Routine screening according to guidelines (typically 3 years).",
    "Atypical Squamous Cells (ASC-US)": "HPV testing recommended. If HPV-positive, colposcopy evaluation.",
    "Low-Grade Squamous Intraepithelial Lesion (LSIL)": "Colposcopy recommended, especially if HPV-positive or persistent.",
    "High-Grade Squamous Intraepithelial Lesion (HSIL)": "Colposcopic evaluation with biopsy and treatment recommended.",
    "Carcinoma": "Immediate referral to gynecologic oncologist for staging and treatment planning.",
}

CLINICAL_EXPLANATIONS = {
    "Normal - NILM": "No epithelial abnormality detected. Cells appear morphologically normal with no signs of dysplasia or malignancy.",
    "Atypical Squamous Cells (ASC-US)": "Atypical squamous cells present with unclear significance. May represent benign reactive changes or early precancerous changes.",
    "Low-Grade Squamous Intraepithelial Lesion (LSIL)": "Low-grade precancerous changes detected. LSIL represents mild dysplasia often associated with HPV infection.",
    "High-Grade Squamous Intraepithelial Lesion (HSIL)": "High-grade precancerous changes detected. HSIL is precancerous and treatable — it is NOT invasive cancer.",
    "Carcinoma": "Features consistent with invasive carcinoma identified. Immediate specialist referral required.",
}

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_INPUT_SIZE = 224

# Normalised mean/std in CHW layout for fast vectorised normalisation
_MEAN_CHW = IMAGE_MEAN[:, None, None]
_STD_CHW  = IMAGE_STD[:, None, None]


def _preprocess_one(image_file) -> np.ndarray:
    """
    Load and preprocess a single image → float32 CHW tensor, no batch dim.
    Staged downsampling keeps memory low on large inputs.
    """
    img = Image.open(image_file)
    w, h = img.size
    max_dim = max(w, h)

    if max_dim > 2048:
        scale = 2048 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)

    img = img.convert("RGB").resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC
    arr = arr.transpose(2, 0, 1)                              # CHW
    arr = (arr - _MEAN_CHW) / _STD_CHW
    return arr                                                 # (3, 224, 224)


class VPSInferenceService:
    """Optimised inference service — singleton, thread-safe."""

    _instance: Optional["VPSInferenceService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "input"
        self.temperature: float = 1.0
        self._n_workers: int = max(2, (os.cpu_count() or 2))

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
        temp_path  = Path(settings.MODEL_DIR) / "temperature.json"

        print(f"[VPS] Loading model: {model_path}")
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            return

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Use all available cores for intra-op parallelism
            n_cores = os.cpu_count() or 2
            opts.intra_op_num_threads = n_cores
            opts.inter_op_num_threads = max(1, n_cores // 2)

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            # Cache input name
            self.input_name = self.session.get_inputs()[0].name
            print(f"[VPS] Model loaded. Input: '{self.input_name}', workers: {self._n_workers}")

            # Warm-up — eliminates slow first-request JIT compilation
            dummy = np.zeros((1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
            self.session.run(None, {self.input_name: dummy})
            print("[VPS] Warm-up done.")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return

        if temp_path.exists():
            try:
                with open(temp_path) as f:
                    self.temperature = float(json.load(f)["temperature"])
                print(f"[VPS] Temperature: T={self.temperature:.4f}")
            except Exception:
                self.temperature = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_batch(self, image_files: List) -> List[Dict]:
        """
        Batch prediction — ONE ONNX session.run() call for all images.
        Images are preprocessed in parallel, then stacked into a single
        tensor before inference.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded.")
        if not image_files:
            return []

        n = len(image_files)
        t0 = time.time()
        print(f"[VPS] Batch of {n} images")

        # ── Step 1: parallel preprocessing ──────────────────────────
        tensors: List[Optional[np.ndarray]] = [None] * n

        def _do_preprocess(idx, f):
            try:
                return idx, _preprocess_one(f)
            except Exception as e:
                print(f"[WARN] Preprocess failed for image {idx}: {e}")
                return idx, None

        workers = min(self._n_workers, n)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_preprocess, i, f): i for i, f in enumerate(image_files)}
            for fut in as_completed(futures):
                idx, arr = fut.result()
                tensors[idx] = arr

        # Separate valid from failed
        valid_indices = [i for i, t in enumerate(tensors) if t is not None]
        if not valid_indices:
            return [None] * n

        # ── Step 2: single batched ONNX inference ────────────────────
        batch = np.stack([tensors[i] for i in valid_indices], axis=0)  # (K, 3, 224, 224)
        t_infer = time.time()
        logits_batch = self.session.run(None, {self.input_name: batch})[0]  # (K, 5)
        print(f"[VPS] ONNX inference ({len(valid_indices)} imgs): {time.time()-t_infer:.3f}s")

        # ── Step 3: decode results ────────────────────────────────────
        results: List[Optional[Dict]] = [None] * n
        for out_idx, img_idx in enumerate(valid_indices):
            results[img_idx] = self._logits_to_result(logits_batch[out_idx])

        print(f"[VPS] Batch total: {time.time()-t0:.3f}s ({(time.time()-t0)/n:.3f}s/img)")
        return results

    def average_predictions(self, predictions_list: List[Dict]) -> Dict:
        """Average probabilities across a list of per-image results."""
        valid = [p for p in predictions_list if p is not None]
        if not valid:
            raise ValueError("No valid predictions to average.")

        avg_probs = np.zeros(5, dtype=np.float32)
        for pred in valid:
            for i in range(5):
                avg_probs[i] += pred["probabilities"].get(CLASS_NAMES[i], 0.0)
        avg_probs /= len(valid)

        predicted_class = int(avg_probs.argmax())
        label = CLASS_NAMES[predicted_class]
        avg_confidence = float(avg_probs.max())
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
        avg_uncertainty = float(entropy / np.log(5))

        confidence_level = "High" if avg_uncertainty < 0.2 else "Moderate" if avg_uncertainty < 0.4 else "Low"
        risk = RISK_LEVELS[predicted_class]

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
            "confidence_interpretation": self._get_confidence_interpretation(avg_confidence, avg_uncertainty),
            "recommendation": RECOMMENDATIONS.get(label, "Consult with a healthcare provider."),
            "explanation": CLINICAL_EXPLANATIONS.get(label, ""),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _logits_to_result(self, logits: np.ndarray) -> Dict:
        scaled = logits / self.temperature
        exp = np.exp(scaled - scaled.max())
        probs = exp / exp.sum()
        predicted_class = int(probs.argmax())
        label = CLASS_NAMES[predicted_class]

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        uncertainty = float(entropy / np.log(5))
        confidence_level = "High" if uncertainty < 0.2 else "Moderate" if uncertainty < 0.4 else "Low"
        risk = RISK_LEVELS[predicted_class]

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
            "confidence_interpretation": self._get_confidence_interpretation(float(probs.max()), uncertainty),
            "recommendation": RECOMMENDATIONS.get(label, "Consult with a healthcare provider."),
            "explanation": CLINICAL_EXPLANATIONS.get(label, ""),
        }

    def _get_confidence_interpretation(self, confidence: float, uncertainty: float) -> Dict:
        if confidence >= 0.80 and uncertainty < 0.3:
            return {
                "level": "HIGH_CONFIDENCE",
                "message": "High Confidence Finding",
                "action": "Standard clinical protocol may be followed",
                "review_required": False,
                "color": "#10b981",
            }
        elif confidence >= 0.60 or uncertainty < 0.5:
            return {
                "level": "MODERATE_CONFIDENCE",
                "message": "Moderate Confidence — Review Recommended",
                "action": "Pathologist review recommended before clinical decisions",
                "review_required": True,
                "color": "#f59e0b",
            }
        else:
            return {
                "level": "LOW_CONFIDENCE",
                "message": "Low Confidence — Pathologist Review Required",
                "action": "MANDATORY: Pathologist review required before clinical decisions",
                "review_required": True,
                "color": "#ef4444",
            }
