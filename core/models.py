from django.db import models
import uuid
from django.contrib.auth import get_user_model

User = get_user_model()


class UserProfile(models.Model):
    """Extended user profile with role and additional fields"""
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('user', 'User'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    phone = models.CharField(max_length=20, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    license_number = models.CharField(max_length=50, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.role}"

    @property
    def full_name(self):
        return self.user.get_full_name() or self.user.username


# Signal to create UserProfile when User is created
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save()
    else:
        UserProfile.objects.create(user=instance)


class Patient(models.Model):
    """Patient model for storing patient information and medical history"""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    patient_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="patients")

    # Basic Information
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField(null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    address = models.TextField(blank=True, null=True)

    # Medical Information
    hpv_status = models.CharField(max_length=20, choices=[('positive', 'Positive'), ('negative', 'Negative'), ('unknown', 'Unknown')], default='unknown')
    last_screening_date = models.DateField(null=True, blank=True)
    pregnancy_status = models.CharField(max_length=20, choices=[('yes', 'Yes'), ('no', 'No'), ('unknown', 'Unknown')], default='unknown')

    # Medical History
    medical_history = models.TextField(blank=True, null=True, help_text="Previous medical conditions, surgeries, etc.")
    medications = models.TextField(blank=True, null=True, help_text="Current medications")
    allergies = models.TextField(blank=True, null=True, help_text="Known allergies")
    family_history = models.TextField(blank=True, null=True, help_text="Family history of relevant conditions")

    # Additional Notes
    notes = models.TextField(blank=True, null=True, help_text="Additional notes or observations")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Patient"
        verbose_name_plural = "Patients"

    def __str__(self):
        return f"{self.full_name} ({self.patient_id})"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def age_calculated(self):
        """Calculate age from date of birth"""
        from django.utils import timezone
        if self.date_of_birth:
            today = timezone.now().date()
            return today.year - self.date_of_birth.year - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        return self.age


class Analysis(models.Model):
    analysis_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    created_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="user_analyses")
    patient = models.ForeignKey('Patient', null=True, blank=True, on_delete=models.SET_NULL, related_name="analyses")
    image = models.ImageField(upload_to="uploads/")
    predicted_class = models.IntegerField()
    predicted_label = models.CharField(max_length=50)
    probabilities = models.JSONField()
    confidence = models.FloatField()
    uncertainty = models.FloatField()
    confidence_level = models.CharField(max_length=20)
    recommendation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.predicted_label} — {self.analysis_id}"
