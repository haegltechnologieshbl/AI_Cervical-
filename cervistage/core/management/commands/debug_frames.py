from django.core.management.base import BaseCommand
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Debug frame generation setup'

    def handle(self, *args, **options):
        self.stdout.write('=== Frame Generation Debug ===\n')

        # Check OpenCV
        try:
            import cv2
            self.stdout.write(self.style.SUCCESS(f'✓ OpenCV installed: cv2 version {cv2.__version__}'))
        except ImportError:
            self.stdout.write(self.style.ERROR('✗ OpenCV NOT installed! Run: pip install opencv-python'))
            return

        # Check MEDIA_ROOT
        self.stdout.write(f'\nMEDIA_ROOT: {settings.MEDIA_ROOT}')
        self.stdout.write(f'MEDIA_URL: {settings.MEDIA_URL}')

        # Check if MEDIA_ROOT exists and is writable
        if os.path.exists(settings.MEDIA_ROOT):
            self.stdout.write(self.style.SUCCESS(f'✓ MEDIA_ROOT exists'))
            if os.access(settings.MEDIA_ROOT, os.W_OK):
                self.stdout.write(self.style.SUCCESS(f'✓ MEDIA_ROOT is writable'))
            else:
                self.stdout.write(self.style.ERROR(f'✗ MEDIA_ROOT is NOT writable'))
        else:
            self.stdout.write(self.style.ERROR(f'✗ MEDIA_ROOT does NOT exist'))

        # Check/create animations directory
        anim_dir = os.path.join(settings.MEDIA_ROOT, 'animations')
        self.stdout.write(f'\nAnimations directory: {anim_dir}')

        try:
            os.makedirs(anim_dir, exist_ok=True)
            self.stdout.write(self.style.SUCCESS(f'✓ Animations directory ready'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Failed to create animations directory: {e}'))

        # Check a recent analysis to see if image path exists
        from core.models import Analysis
        recent = Analysis.objects.filter(image__isnull=False).order_by('-created_at').first()
        if recent:
            self.stdout.write(f'\nRecent analysis: {recent.analysis_id}')
            self.stdout.write(f'Image path: {recent.image.path if recent.image else "None"}')
            self.stdout.write(f'Image URL: {recent.image.url if recent.image else "None"}')

            if recent.image and os.path.exists(recent.image.path):
                self.stdout.write(self.style.SUCCESS(f'✓ Image file exists on disk'))
            else:
                self.stdout.write(self.style.ERROR(f'✗ Image file NOT found on disk'))
        else:
            self.stdout.write(self.style.WARNING('No analyses found to check'))

        # Check if any frames exist
        if os.path.exists(anim_dir):
            frame_count = sum(len(files) for _, _, files in os.walk(anim_dir) if files)
            self.stdout.write(f'\nExisting frames in animations/: {frame_count}')
        else:
            self.stdout.write(self.style.WARNING('\nNo animations directory found'))
