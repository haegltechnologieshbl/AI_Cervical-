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
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="patients")

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

    # ========== HOSPITAL MANAGEMENT ENHANCEMENT FIELDS ==========

    # Personal Information Extensions
    marital_status = models.CharField(max_length=20, choices=[
        ('single', 'Single'),
        ('married', 'Married'),
        ('widowed', 'Widowed'),
        ('divorced', 'Divorced'),
        ('separated', 'Separated')
    ], null=True, blank=True, help_text="Patient's marital status")

    occupation = models.CharField(max_length=100, null=True, blank=True, help_text="Patient's occupation")

    education_level = models.CharField(max_length=50, choices=[
        ('none', 'None'),
        ('primary', 'Primary'),
        ('secondary', 'Secondary'),
        ('graduate', 'Graduate'),
        ('post_graduate', 'Post Graduate')
    ], null=True, blank=True, help_text="Patient's highest education level")

    blood_group = models.CharField(max_length=5, choices=[
        ('A+', 'A+'), ('A-', 'A-'),
        ('B+', 'B+'), ('B-', 'B-'),
        ('O+', 'O+'), ('O-', 'O-'),
        ('AB+', 'AB+'), ('AB-', 'AB-')
    ], null=True, blank=True, help_text="Patient's blood group")

    # Enhanced Contact Information
    alternate_contact = models.CharField(max_length=20, null=True, blank=True, help_text="Alternate contact number")

    district = models.CharField(max_length=100, null=True, blank=True, help_text="District of residence")

    state = models.CharField(max_length=100, null=True, blank=True, help_text="State of residence")

    pin_code = models.CharField(max_length=10, null=True, blank=True, help_text="PIN/Postal code")

    # Identification Details
    aadhaar_number = models.CharField(max_length=12, null=True, blank=True, unique=True, help_text="Aadhaar number (12 digits)")

    abha_health_id = models.CharField(max_length=50, null=True, blank=True, unique=True, help_text="ABHA Health ID number")

    medical_record_number = models.CharField(max_length=50, null=True, blank=True, unique=True, help_text="Medical Record Number")

    # Emergency Contact Information
    emergency_contact_name = models.CharField(max_length=100, null=True, blank=True, help_text="Emergency contact person name")

    emergency_contact_relationship = models.CharField(max_length=50, null=True, blank=True, help_text="Relationship with emergency contact")

    emergency_contact_number = models.CharField(max_length=20, null=True, blank=True, help_text="Emergency contact phone number")

    # Consent Tracking
    consent_screening = models.BooleanField(default=False, help_text="Consent given for screening")

    consent_image_capture = models.BooleanField(default=False, help_text="Consent given for image capture")

    consent_ai_analysis = models.BooleanField(default=False, help_text="Consent given for AI analysis")

    digital_signature = models.TextField(null=True, blank=True, help_text="Digital signature (base64 encoded)")

    # Medical Record Extensions - FIGO Staging
    current_figo_stage = models.CharField(max_length=20, choices=[
        ('0', 'Stage 0 - Pre-cancer'),
        ('IA1', 'Stage IA1'),
        ('IA2', 'Stage IA2'),
        ('IB1', 'Stage IB1'),
        ('IB2', 'Stage IB2'),
        ('IIA', 'Stage IIA'),
        ('IIB', 'Stage IIB'),
        ('IIIA', 'Stage IIIA'),
        ('IIIB', 'Stage IIIB'),
        ('IVA', 'Stage IVA'),
        ('IVB', 'Stage IVB')
    ], null=True, blank=True, help_text="Current FIGO staging if diagnosed")

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


# ========== HOSPITAL MANAGEMENT MODELS ==========

class GynecologicalHistory(models.Model):
    """Gynecological and obstetric history for female patients"""
    history_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.OneToOneField(Patient, on_delete=models.CASCADE, related_name='gyn_history')

    # Menstrual History
    menarche_age = models.IntegerField(null=True, blank=True, help_text="Age at first menstruation")
    menopause_age = models.IntegerField(null=True, blank=True, help_text="Age at menopause")
    menstrual_history = models.TextField(null=True, blank=True, help_text="Menstrual pattern details (regularity, flow, etc.)")
    last_pap_smear = models.DateField(null=True, blank=True, help_text="Date of last Pap smear test")
    last_pap_result = models.CharField(max_length=50, null=True, blank=True, help_text="Result of last Pap smear")

    # Contraceptive History
    contraceptive_use = models.TextField(null=True, blank=True, help_text="Current/past contraceptive methods")

    # Sexual History (confidential)
    sexual_history = models.TextField(null=True, blank=True, help_text="Relevant sexual history (if disclosed)")

    # Other Gynecological Information
    last_colposcopy = models.DateField(null=True, blank=True, help_text="Date of last colposcopy")
    colposcopy_findings = models.TextField(null=True, blank=True, help_text="Colposcopy findings")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Gynecological History"
        verbose_name_plural = "Gynecological Histories"

    def __str__(self):
        return f"{self.patient.full_name} - Gyn History"


class ReproductiveHistory(models.Model):
    """Reproductive and pregnancy history"""
    history_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.OneToOneField(Patient, on_delete=models.CASCADE, related_name='repro_history')

    # Pregnancy Counts
    pregnancies_count = models.IntegerField(default=0, null=True, blank=True, help_text="Total number of pregnancies")
    deliveries_count = models.IntegerField(default=0, null=True, blank=True, help_text="Total number of deliveries")
    abortions_count = models.IntegerField(default=0, null=True, blank=True, help_text="Total number of abortions/miscarriages")
    live_births = models.IntegerField(default=0, null=True, blank=True, help_text="Number of live births")

    # Pregnancy Details
    complications = models.TextField(null=True, blank=True, help_text="Pregnancy/delivery complications")

    # Reproductive Health
    infertility_history = models.TextField(null=True, blank=True, help_text="History of infertility treatments")
    fertility_treatments = models.TextField(null=True, blank=True, help_text="Fertility treatments undergone")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Reproductive History"
        verbose_name_plural = "Reproductive Histories"

    def __str__(self):
        return f"{self.patient.full_name} - Reproductive History"


class ScreeningRecord(models.Model):
    """Historical screening records from other facilities"""
    record_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='screening_records')

    screening_date = models.DateField(help_text="Date of screening")

    screening_type = models.CharField(max_length=50, choices=[
        ('pap_smear', 'Pap Smear'),
        ('hpv_test', 'HPV Test'),
        ('hpv_vaccination', 'HPV Vaccination'),
        ('colposcopy', 'Colposcopy'),
        ('biopsy', 'Biopsy'),
        ('visual_inspection', 'Visual Inspection with Acetic Acid'),
        ('mri', 'MRI Scan'),
        ('ct_scan', 'CT Scan'),
        ('ultrasound', 'Ultrasound'),
        ('other', 'Other'),
    ], help_text="Type of screening procedure")

    facility_name = models.CharField(max_length=200, help_text="Name of facility where screening was done")

    result = models.TextField(help_text="Screening result details")

    findings = models.TextField(null=True, blank=True, help_text="Additional findings from screening")

    uploaded_report = models.ImageField(upload_to='screening_reports/', null=True, blank=True, help_text="Uploaded screening report image")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-screening_date']
        verbose_name = "Screening Record"
        verbose_name_plural = "Screening Records"

    def __str__(self):
        return f"{self.patient.full_name} - {self.screening_type} on {self.screening_date}"


class FollowUpRecord(models.Model):
    """Patient follow-up tracking and scheduling"""
    followup_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='follow_ups')

    scheduled_date = models.DateField(help_text="Date follow-up is scheduled")

    completed_date = models.DateField(null=True, blank=True, help_text="Date follow-up was completed")

    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('missed', 'Missed'),
        ('cancelled', 'Cancelled'),
        ('rescheduled', 'Rescheduled'),
    ], default='pending', help_text="Current status of follow-up")

    purpose = models.CharField(max_length=200, help_text="Purpose of follow-up (e.g., 'Review Pap results', 'Post-treatment check')")

    notes = models.TextField(null=True, blank=True, help_text="Additional notes about follow-up")

    findings = models.TextField(null=True, blank=True, help_text="Findings from follow-up visit")

    next_action = models.TextField(null=True, blank=True, help_text="Recommended next action after follow-up")

    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_followups')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['scheduled_date']
        verbose_name = "Follow-up Record"
        verbose_name_plural = "Follow-up Records"

    def __str__(self):
        return f"{self.patient.full_name} - Follow-up on {self.scheduled_date}"


class TreatmentRecord(models.Model):
    """Patient treatment history and records"""
    treatment_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='treatments')

    treatment_type = models.CharField(max_length=100, choices=[
        ('observation', 'Observation'),
        ('medication', 'Medication'),
        ('surgery', 'Surgery'),
        ('chemotherapy', 'Chemotherapy'),
        ('radiation', 'Radiation Therapy'),
        ('cryotherapy', 'Cryotherapy'),
        ('leep', 'LEEP Procedure'),
        ('conization', 'Cold Knife Conization'),
        ('hysterectomy', 'Hysterectomy'),
        ('trachelectomy', 'Trachelectomy'),
        ('immunotherapy', 'Immunotherapy'),
        ('palliative', 'Palliative Care'),
        ('other', 'Other'),
    ], help_text="Type of treatment")

    start_date = models.DateField(help_text="Treatment start date")

    end_date = models.DateField(null=True, blank=True, help_text="Treatment end date (if applicable)")

    description = models.TextField(help_text="Treatment details and protocol")

    outcome = models.TextField(null=True, blank=True, help_text="Treatment outcome/results")

    side_effects = models.TextField(null=True, blank=True, help_text="Side effects experienced")

    prescribing_doctor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='prescribed_treatments')

    hospital_facility = models.CharField(max_length=200, null=True, blank=True, help_text="Facility where treatment was given")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-start_date']
        verbose_name = "Treatment Record"
        verbose_name_plural = "Treatment Records"

    def __str__(self):
        return f"{self.patient.full_name} - {self.get_treatment_type_display()}"


class Appointment(models.Model):
    """Patient appointments and scheduling"""
    appointment_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='appointments')

    scheduled_date = models.DateTimeField(help_text="Date and time of appointment")

    appointment_type = models.CharField(max_length=50, choices=[
        ('screening', 'Screening Appointment'),
        ('follow_up', 'Follow-up Appointment'),
        ('consultation', 'Consultation'),
        ('treatment', 'Treatment Session'),
        ('colposcopy', 'Colposcopy'),
        ('biopsy', 'Biopsy'),
        ('post_op', 'Post-Operative Check'),
        ('emergency', 'Emergency Visit'),
        ('telehealth', 'Telehealth Consultation'),
    ], help_text="Type of appointment")

    status = models.CharField(max_length=20, choices=[
        ('scheduled', 'Scheduled'),
        ('confirmed', 'Confirmed'),
        ('checked_in', 'Checked In'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
    ], default='scheduled', help_text="Appointment status")

    priority = models.CharField(max_length=20, choices=[
        ('routine', 'Routine'),
        ('urgent', 'Urgent'),
        ('emergency', 'Emergency'),
    ], default='routine', help_text="Appointment priority")

    notes = models.TextField(null=True, blank=True, help_text="Additional notes for appointment")

    reminder_sent = models.BooleanField(default=False, help_text="Whether appointment reminder was sent")

    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_appointments')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['scheduled_date']
        verbose_name = "Appointment"
        verbose_name_plural = "Appointments"

    def __str__(self):
        return f"{self.patient.full_name} - {self.get_appointment_type_display()} on {self.scheduled_date}"


class RiskAssessment(models.Model):
    """Patient risk assessment and scoring for cervical cancer risk"""
    assessment_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='risk_assessments')

    # Overall Risk Score (0-100)
    risk_score = models.IntegerField(default=0, help_text="Overall risk score from 0-100")

    risk_level = models.CharField(max_length=20, choices=[
        ('low', 'Low Risk'),
        ('moderate', 'Moderate Risk'),
        ('high', 'High Risk'),
        ('very_high', 'Very High Risk'),
    ], help_text="Overall risk level assessment")

    risk_factors = models.JSONField(default=dict, help_text="Identified risk factors with scores")

    recommendations = models.TextField(help_text="Clinical recommendations based on risk assessment")

    next_review_date = models.DateField(null=True, blank=True, help_text="Date for next risk assessment review")

    assessed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='assessments')

    # ===== RISK FACTORS =====

    # Age Risk
    age_risk = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
    ], null=True, blank=True)

    # HPV Status
    hpv_risk = models.CharField(max_length=20, choices=[
        ('negative', 'Negative'),
        ('positive_low_risk', 'Positive (Low Risk HPV)'),
        ('positive_high_risk', 'Positive (High Risk HPV)'),
        ('unknown', 'Unknown'),
    ], null=True, blank=True)

    # Smoking History
    smoking_status = models.CharField(max_length=20, choices=[
        ('non_smoker', 'Non-smoker'),
        ('former_smoker', 'Former Smoker'),
        ('current_smoker', 'Current Smoker'),
    ], null=True, blank=True)

    smoking_years = models.IntegerField(null=True, blank=True, help_text="Years of smoking")

    # Alcohol Use
    alcohol_use = models.CharField(max_length=20, choices=[
        ('none', 'None'),
        ('occasional', 'Occasional'),
        ('moderate', 'Moderate'),
        ('heavy', 'Heavy'),
    ], null=True, blank=True)

    # Family History of Cancer
    family_history_risk = models.BooleanField(default=False, help_text="Family history of cervical cancer")
    family_history_details = models.TextField(null=True, blank=True, help_text="Details of family history")

    # HIV Status
    hiv_status = models.CharField(max_length=20, choices=[
        ('negative', 'Negative'),
        ('positive', 'Positive'),
        ('unknown', 'Unknown'),
    ], null=True, blank=True)

    # Diabetes
    diabetes = models.BooleanField(default=False, help_text="Patient has diabetes")
    diabetes_type = models.CharField(max_length=20, choices=[
        ('type1', 'Type 1'),
        ('type2', 'Type 2'),
        ('gestational', 'Gestational'),
    ], null=True, blank=True)

    # Pregnancy History
    number_of_pregnancies = models.IntegerField(null=True, blank=True, help_text="Number of pregnancies")
    age_at_first_pregnancy = models.IntegerField(null=True, blank=True, help_text="Age at first pregnancy")
    age_at_first_sexual_intercourse = models.IntegerField(null=True, blank=True, help_text="Age at first sexual intercourse")

    # Oral Contraceptive Use
    oral_contraceptive_use = models.CharField(max_length=20, choices=[
        ('never', 'Never'),
        ('past', 'Past User'),
        ('current', 'Current User'),
    ], null=True, blank=True)

    oral_contraceptive_years = models.IntegerField(null=True, blank=True, help_text="Years of oral contraceptive use")

    # Sexual History
    sexual_partners_count = models.IntegerField(null=True, blank=True, help_text="Number of lifetime sexual partners")
    age_at_first_sexual_intercourse = models.IntegerField(null=True, blank=True, help_text="Age at first sexual intercourse")

    # Other Risk Factors
    immunocompromised = models.BooleanField(default=False, help_text="Patient is immunocompromised (other than HIV)")
    immunocompromised_details = models.TextField(null=True, blank=True, help_text="Details of immunocompromised status")

    symptoms_risk = models.CharField(max_length=30, choices=[
        ('asymptomatic', 'Asymptomatic'),
        ('symptomatic_benign', 'Symptomatic (likely benign)'),
        ('symptomatic_concerning', 'Symptomatic (concerning)'),
    ], null=True, blank=True)

    # Previous Abnormal Pap Smears
    previous_abnormal_pap = models.BooleanField(default=False, help_text="Previous abnormal Pap smear results")
    previous_abnormal_pap_details = models.TextField(null=True, blank=True, help_text="Details of previous abnormal results")

    # Vaccination Status
    hpv_vaccinated = models.BooleanField(default=False, help_text="HPV vaccinated")
    hpv_vaccine_doses = models.IntegerField(null=True, blank=True, help_text="Number of HPV vaccine doses received")

    # Screening History
    last_pap_smear_date = models.DateField(null=True, blank=True, help_text="Date of last Pap smear")
    last_pap_smear_result = models.CharField(max_length=50, null=True, blank=True, help_text="Result of last Pap smear")

    # AI Confidence Score (if linked to analysis)
    ai_confidence_score = models.FloatField(null=True, blank=True, help_text="AI confidence score from latest analysis")

    # Screening Recommendation
    screening_recommendation = models.CharField(max_length=20, choices=[
        ('routine', 'Routine Screening'),
        ('annual', 'Annual Screening'),
        ('semi_annual', 'Semi-annual (6 months)'),
        ('quarterly', 'Quarterly (3 months)'),
        ('immediate', 'Immediate Follow-up'),
    ], default='routine')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Risk Assessment"
        verbose_name_plural = "Risk Assessments"

    def __str__(self):
        return f"{self.patient.full_name} - Risk Score: {self.risk_score} ({self.get_risk_level_display()})"


# ========== END HOSPITAL MANAGEMENT MODELS ==========

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
