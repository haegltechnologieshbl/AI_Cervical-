"""
Image Quality Validator for Cervical Cytology Samples

This module provides validation to ensure uploaded images are likely
to be valid cervical cytology/cell sample images before AI analysis.

Medical AI systems should analyze appropriate images to avoid:
- False positives/negatives on invalid inputs
- Misleading results on non-medical images
- Wasted computational resources
"""

import io
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
import cv2


class ImageValidationError:
    """Error codes and messages for image validation failures."""

    INVALID_BRIGHTNESS = (
        "invalid_brightness",
        "Image brightness is outside acceptable range for cytology samples. "
        "This may indicate an overexposed or underexposed image."
    )

    LOW_COLOR_VARIANCE = (
        "low_variance",
        "Image lacks color variation typical of cervical cell samples. "
        "This may be a solid color, drawing, or document."
    )

    LOW_SATURATION = (
        "low_saturation",
        "Low color saturation detected. Cervical cytology samples typically "
        "have stained cells with rich coloration."
    )

    UNIFORM_TEXTURE = (
        "uniform_texture",
        "Image has uniform texture without cellular structures. "
        "This may not be a microscopy image."
    )

    ASPECT_RATIO_UNUSUAL = (
        "unusual_aspect",
        "Image aspect ratio is unusual for cytology samples."
    )

    TOO_SMALL = (
        "too_small",
        "Image resolution is too low for reliable analysis."
    )

    VALID = ("valid", "Image passes all quality checks for cytology analysis.")


class ImageValidator:
    """
    Validates images for cervical cytology analysis.

    Uses multiple heuristic checks to determine if an image is likely
    to be a valid cervical cell sample:
    1. Brightness/contrast checks
    2. Color variance analysis
    3. Saturation analysis (for stained samples)
    4. Edge density (for cellular structures)
    5. Aspect ratio checks
    """

    # Thresholds for validation (based on typical cytology images)
    MIN_BRIGHTNESS = 40      # Minimum average pixel brightness
    MAX_BRIGHTNESS = 220     # Maximum average pixel brightness
    MIN_COLOR_STD = 25       # Minimum standard deviation of colors
    MIN_SATURATION = 20      # Minimum average saturation (0-100 scale)
    MIN_EDGE_DENSITY = 0.02  # Minimum ratio of edge pixels to total pixels
    MIN_WIDTH = 200          # Minimum image width
    MIN_HEIGHT = 200         # Minimum image height
    MAX_ASPECT_RATIO = 3.0   # Maximum width:height or height:width ratio

    @classmethod
    def validate_file(cls, image_file) -> Dict:
        """
        Validate an uploaded image file.

        Args:
            image_file: Django UploadedFile or file-like object

        Returns:
            Dict with keys:
                - is_valid (bool): Overall validation result
                - score (float): Quality score 0-100
                - errors (list): List of (code, message) tuples
                - warnings (list): List of warning messages
                - metrics (dict): Computed image metrics
        """
        result = {
            "is_valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

        try:
            # Read and reset file pointer
            image_file.seek(0)
            img_bytes = image_file.read()
            image_file.seek(0)

            # Open with PIL
            img = Image.open(io.BytesIO(img_bytes))

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            arr = np.array(img)

            # Run all validation checks
            cls._check_basic_properties(img, arr, result)
            cls._check_brightness_contrast(arr, result)
            cls._check_color_variance(arr, result)
            cls._check_saturation(arr, result)
            cls._check_cellular_structure(arr, result)
            cls._calculate_quality_score(result)

        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(("processing_error", f"Failed to validate image: {str(e)}"))
            result["score"] = 0.0

        return result

    @staticmethod
    def _check_basic_properties(img: Image.Image, arr: np.ndarray, result: Dict) -> None:
        """Check basic image properties."""
        width, height = img.size
        result["metrics"]["width"] = width
        result["metrics"]["height"] = height
        result["metrics"]["aspect_ratio"] = width / height if height > 0 else 0

        # Check minimum size
        if width < ImageValidator.MIN_WIDTH or height < ImageValidator.MIN_HEIGHT:
            result["is_valid"] = False
            result["errors"].append(ImageValidationError.TOO_SMALL)

        # Check aspect ratio
        aspect = width / height if height > 0 else 0
        if aspect > ImageValidator.MAX_ASPECT_RATIO or aspect < (1 / ImageValidator.MAX_ASPECT_RATIO):
            result["warnings"].append(ImageValidationError.ASPECT_RATIO_UNUSUAL[1])

    @staticmethod
    def _check_brightness_contrast(arr: np.ndarray, result: Dict) -> None:
        """Check image brightness and contrast."""
        brightness = arr.mean()
        result["metrics"]["brightness"] = round(brightness, 2)

        if brightness < ImageValidator.MIN_BRIGHTNESS:
            result["is_valid"] = False
            result["errors"].append(ImageValidationError.INVALID_BRIGHTNESS)
            result["warnings"].append(f"Image too dark (brightness: {brightness:.1f}, minimum: {ImageValidator.MIN_BRIGHTNESS})")
        elif brightness > ImageValidator.MAX_BRIGHTNESS:
            result["is_valid"] = False
            result["errors"].append(ImageValidationError.INVALID_BRIGHTNESS)
            result["warnings"].append(f"Image too bright (brightness: {brightness:.1f}, maximum: {ImageValidator.MAX_BRIGHTNESS})")

        # Calculate contrast as standard deviation
        contrast = arr.std()
        result["metrics"]["contrast"] = round(contrast, 2)

    @staticmethod
    def _check_color_variance(arr: np.ndarray, result: Dict) -> None:
        """Check color variance across RGB channels."""
        color_std = arr.std(axis=(0, 1))  # Std for each RGB channel
        avg_color_std = color_std.mean()
        result["metrics"]["color_std"] = round(avg_color_std, 2)
        result["metrics"]["r_std"] = round(float(color_std[0]), 2)
        result["metrics"]["g_std"] = round(float(color_std[1]), 2)
        result["metrics"]["b_std"] = round(float(color_std[2]), 2)

        if avg_color_std < ImageValidator.MIN_COLOR_STD:
            result["is_valid"] = False
            result["errors"].append(ImageValidationError.LOW_COLOR_VARIANCE)
            result["warnings"].append(
                f"Low color variance (std: {avg_color_std:.1f}, minimum: {ImageValidator.MIN_COLOR_STD}). "
                "This may be a solid color image, drawing, or document."
            )

    @staticmethod
    def _check_saturation(arr: np.ndarray, result: Dict) -> None:
        """Check color saturation (for Pap stain detection)."""
        # Convert to HSV
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0 * 100  # Scale to 0-100
        result["metrics"]["saturation"] = round(saturation, 2)

        if saturation < ImageValidator.MIN_SATURATION:
            result["warnings"].append(
                f"Low saturation ({saturation:.1f}%). Cervical cytology samples typically "
                f"have stained cells with higher saturation."
            )

            # Only fail if saturation is extremely low (likely grayscale/b&w)
            if saturation < 10:
                result["is_valid"] = False
                result["errors"].append(ImageValidationError.LOW_SATURATION)

    @staticmethod
    def _check_cellular_structure(arr: np.ndarray, result: Dict) -> None:
        """Check for cellular structures using edge detection."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (255 * edges.shape[0] * edges.shape[1])

        result["metrics"]["edge_density"] = round(edge_density, 4)

        if edge_density < ImageValidator.MIN_EDGE_DENSITY:
            result["warnings"].append(
                f"Low edge density ({edge_density:.4f}). Image may lack cellular structures."
            )

    @staticmethod
    def _calculate_quality_score(result: Dict) -> None:
        """Calculate overall quality score from validation metrics."""
        score = 100.0

        # Deduct points for each warning
        warning_deduction = 15
        score -= len(result["warnings"]) * warning_deduction

        # Major deductions for errors
        error_deduction = 30
        score -= len(result["errors"]) * error_deduction

        result["score"] = max(0.0, round(score, 1))

    @staticmethod
    def get_validation_summary(validation_result: Dict) -> str:
        """
        Generate a human-readable validation summary.

        Returns a string suitable for display to users.
        """
        if validation_result["is_valid"]:
            return f"✓ Image quality check passed (Score: {validation_result['score']}/100)"

        errors = validation_result["errors"]
        warnings = validation_result["warnings"]

        messages = []
        for code, msg in errors:
            messages.append(f"❌ {msg}")

        for warning in warnings:
            messages.append(f"⚠️ {warning}")

        return "\n".join(messages) if messages else "Image validation failed"


# Convenience function for quick validation
def validate_image(image_file) -> Tuple[bool, str, float]:
    """
    Quick validation of an uploaded image.

    Returns:
        Tuple of (is_valid, message, quality_score)
    """
    result = ImageValidator.validate_file(image_file)
    message = ImageValidator.get_validation_summary(result)
    return result["is_valid"], message, result["score"]
