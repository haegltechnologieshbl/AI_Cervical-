from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # Load the ONNX model once at startup
        from services.inference import InferenceService
        InferenceService.get()
