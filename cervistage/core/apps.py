from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        from services.inference_vps import VPSInferenceService
        VPSInferenceService.get()
