from django.db import models
import uuid
from django.contrib.auth import get_user_model

User = get_user_model()


class UserProfile(models.Model):
    """Extended user profile with role and additional fields"""
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('senior_technician', 'Senior Technician'),
        ('user', 'Doctor/User'),
        ('patient', 'Patient'),
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
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="patients")
    
    # Workflow assignments
    assigned_doctor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="assigned_patients_doctor", limit_choices_to={'profile__role': 'user'})
    assigned_technician = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="assigned_patients_technician", limit_choices_to={'profile__role': 'senior_technician'})


    # Basic Information
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField(null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    user_account = models.OneToOneField(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='patient_record', help_text="Linked user account for patient login")
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


class PapSmearTest(models.Model):
    """Specific model for Lab Technicians to record Pap smear results"""
    test_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='pap_smear_tests')
    technician = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='performed_pap_smears')
    
    result = models.CharField(max_length=20, choices=[
        ('+ve', 'Positive (+ve)'),
        ('-ve', 'Negative (-ve)'),
    ], help_text="Pap smear test result")
    
    notes = models.TextField(blank=True, null=True, help_text="Additional technician notes")
    test_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-test_date']

    def __str__(self):
        return f"Pap Smear {self.result} for {self.patient.full_name}"


class CheckupRecommendation(models.Model):
    """Specific model for Doctors to add checkups after reviewing Pap smear"""
    checkup_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='checkup_recommendations')
    doctor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='prescribed_checkups')
    pap_smear_test = models.ForeignKey(PapSmearTest, on_delete=models.SET_NULL, null=True, blank=True, related_name='associated_checkups')
    
    # Recommended follow-ups/tests
    hpv_pcr_testing = models.BooleanField(default=False, help_text="Recommend HPV PCR Testing")
    colposcopy = models.BooleanField(default=False, help_text="Recommend Colposcopy")
    histopathology = models.BooleanField(default=False, help_text="Recommend Histopathology (Gold Standard)")
    mri_scan = models.BooleanField(default=False, help_text="Recommend MRI (only for confirmed invasive cancer)")
    treatment_follow_up = models.TextField(blank=True, null=True, help_text="Treatment & Follow-up plan")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

    def recommended_list(self):
        """Human-readable list of the tests the doctor recommended."""
        labels = []
        if self.hpv_pcr_testing:
            labels.append("HPV PCR")
        if self.colposcopy:
            labels.append("Colposcopy")
        if self.histopathology:
            labels.append("Histopathology")
        if self.mri_scan:
            labels.append("MRI Scan")
        return labels

    def __str__(self):
        return f"Checkups for {self.patient.full_name} by {self.doctor}"


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
    recommended_checkups = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Batch metadata for multi-image analyses
    batch_id = models.UUIDField(null=True, blank=True, db_index=True, help_text="ID of the batch this analysis belongs to")
    is_batch_primary = models.BooleanField(default=False, help_text="Whether this is the primary analysis showing aggregated results")
    batch_total_count = models.IntegerField(null=True, blank=True, help_text="Total number of images in the batch")
    batch_position = models.IntegerField(null=True, blank=True, help_text="Position of this image in the batch (1-indexed)")
    batch_aggregated_data = models.JSONField(null=True, blank=True, help_text="Aggregated results for batch primary analysis")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.predicted_label} — {self.analysis_id}"


class ClinicalNote(models.Model):
    """Clinical notes added by doctors or technicians on a patient visit."""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='clinical_notes')
    analysis = models.ForeignKey(Analysis, on_delete=models.SET_NULL, null=True, blank=True, related_name='clinical_notes')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='clinical_notes')
    note = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Note by {self.created_by} on {self.patient} ({self.created_at:%Y-%m-%d})"


class Notification(models.Model):
    """In-app notification shown in the navbar bell."""
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.CharField(max_length=255)
    url = models.CharField(max_length=255, blank=True, default='')
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"To {self.recipient}: {self.message}"

    @classmethod
    def notify(cls, recipient, message, url=''):
        """Create a notification. Silently no-ops if there is no recipient."""
        if recipient is None:
            return None
        return cls.objects.create(recipient=recipient, message=message, url=url or '')


class Feedback(models.Model):
    """Feedback from doctors and patients about the platform."""

    FEEDBACK_TYPE_CHOICES = [
        ('complaint', 'Complaint'),
        ('suggestion', 'Suggestion'),
        ('bug_report', 'Bug Report'),
        ('appreciation', 'Appreciation'),
        ('other', 'Other'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('reviewed', 'Reviewed'),
        ('resolved', 'Resolved'),
    ]

    RATING_CHOICES = [
        (1, '★☆☆☆☆'),
        (2, '★★☆☆☆'),
        (3, '★★★☆☆'),
        (4, '★★★★☆'),
        (5, '★★★★★'),
    ]

    feedback_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    submitted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='submitted_feedback')

    # Core feedback fields
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES, default='suggestion')
    subject = models.CharField(max_length=200)
    message = models.TextField()
    rating = models.IntegerField(choices=RATING_CHOICES, null=True, blank=True, help_text="Star rating (1-5)")

    # Optional related entities
    related_analysis = models.ForeignKey('Analysis', on_delete=models.SET_NULL, null=True, blank=True, related_name='feedback')
    related_patient = models.ForeignKey('Patient', on_delete=models.SET_NULL, null=True, blank=True, related_name='feedback')

    # Admin response fields
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    admin_notes = models.TextField(blank=True, null=True, help_text="Admin response or notes")
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviewed_feedback')
    reviewed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Feedback'
        verbose_name_plural = 'Feedback'

    def __str__(self):
        return f"{self.feedback_type} - {self.subject}"
