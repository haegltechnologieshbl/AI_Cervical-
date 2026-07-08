from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("", include("cervistage.core.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
