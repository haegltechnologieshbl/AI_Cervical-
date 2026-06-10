import re
from rest_framework import serializers
from .models import (
    Analysis, Patient, GynecologicalHistory, ReproductiveHistory,
    ScreeningRecord, FollowUpRecord, TreatmentRecord, Appointment, RiskAssessment
)


class PatientSerializer(serializers.ModelSerializer):
    """Comprehensive serializer for Patient model with all fields"""
    full_name = serializers.SerializerMethodField()
    age_calculated = serializers.SerializerMethodField()
    analyses_count = serializers.SerializerMethodField()
    latest_figo_stage = serializers.CharField(source='current_figo_stage', read_only=True)

    class Meta:
        model = Patient
        fields = [
            # Existing fields
            'patient_id', 'first_name', 'last_name', 'full_name',
            'date_of_birth', 'age', 'age_calculated', 'gender',
            'phone', 'email', 'address',
            'hpv_status', 'last_screening_date', 'pregnancy_status',
            'medical_history', 'medications', 'allergies', 'family_history',
            'notes', 'created_at', 'updated_at', 'analyses_count',

            # New personal fields
            'marital_status', 'occupation', 'education_level', 'blood_group',

            # New contact fields
            'alternate_contact', 'district', 'state', 'pin_code',

            # New identification fields
            'aadhaar_number', 'abha_health_id', 'medical_record_number',

            # New emergency fields
            'emergency_contact_name', 'emergency_contact_relationship',
            'emergency_contact_number',

            # New consent fields
            'consent_screening', 'consent_image_capture', 'consent_ai_analysis',
            'digital_signature',

            # New medical fields
            'current_figo_stage', 'latest_figo_stage',
        ]
        read_only_fields = ['patient_id', 'created_at', 'updated_at']

    def get_full_name(self, obj):
        return obj.full_name

    def get_age_calculated(self, obj):
        return obj.age_calculated

    def get_analyses_count(self, obj):
        return obj.analyses.count()

    def validate_aadhaar_number(self, value):
        """Validate Aadhaar number format (12 digits)"""
        if value:
            value = str(value).strip()
            if not re.match(r'^\d{12}$', value):
                raise serializers.ValidationError("Aadhaar number must be 12 digits")
        return value

    def validate_pin_code(self, value):
        """Validate PIN code format (6 digits)"""
        if value:
            value = str(value).strip()
            if not re.match(r'^\d{6}$', value):
                raise serializers.ValidationError("PIN code must be 6 digits")
        return value


class PatientListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for patient list views"""
    full_name = serializers.SerializerMethodField()
    analyses_count = serializers.SerializerMethodField()
    last_analysis_date = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = [
            'patient_id', 'full_name', 'age', 'gender',
            'phone', 'hpv_status', 'analyses_count', 'last_analysis_date',
            'marital_status', 'district', 'current_figo_stage'
        ]

    def get_full_name(self, obj):
        return obj.full_name

    def get_analyses_count(self, obj):
        return obj.analyses.count()

    def get_last_analysis_date(self, obj):
        last_analysis = obj.analyses.first()
        return last_analysis.created_at if last_analysis else None


class AnalysisSerializer(serializers.ModelSerializer):
    """Serializer for Analysis model with patient information"""
    patient_name = serializers.SerializerMethodField()
    patient_id = serializers.SerializerMethodField()

    class Meta:
        model = Analysis
        fields = "__all__"
        read_only_fields = [
            "analysis_id", "predicted_class", "predicted_label",
            "probabilities", "confidence", "uncertainty",
            "confidence_level", "recommendation", "created_at",
        ]

    def get_patient_name(self, obj):
        if obj.patient:
            return obj.patient.full_name
        return None

    def get_patient_id(self, obj):
        if obj.patient:
            return str(obj.patient.patient_id)
        return None


class AnalyzeRequestSerializer(serializers.Serializer):
    file = serializers.ListField(child=serializers.ImageField(), required=False)

    def validate(self, data):
        # Support single or multiple file uploads
        if 'file' not in data or not data['file']:
            raise serializers.ValidationError("At least one image file is required.")
        return data


# ========== HOSPITAL MANAGEMENT SERIALIZERS ==========

class GynecologicalHistorySerializer(serializers.ModelSerializer):
    """Serializer for Gynecological History model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)

    class Meta:
        model = GynecologicalHistory
        fields = '__all__'
        read_only_fields = ['history_id', 'created_at', 'updated_at']


class ReproductiveHistorySerializer(serializers.ModelSerializer):
    """Serializer for Reproductive History model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)

    class Meta:
        model = ReproductiveHistory
        fields = '__all__'
        read_only_fields = ['history_id', 'created_at', 'updated_at']


class ScreeningRecordSerializer(serializers.ModelSerializer):
    """Serializer for Screening Record model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)

    class Meta:
        model = ScreeningRecord
        fields = '__all__'
        read_only_fields = ['record_id', 'created_at']


class FollowUpRecordSerializer(serializers.ModelSerializer):
    """Serializer for Follow-Up Record model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)

    class Meta:
        model = FollowUpRecord
        fields = '__all__'
        read_only_fields = ['followup_id', 'created_at', 'updated_at']


class TreatmentRecordSerializer(serializers.ModelSerializer):
    """Serializer for Treatment Record model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    prescribing_doctor_name = serializers.CharField(source='prescribing_doctor.username', read_only=True)

    class Meta:
        model = TreatmentRecord
        fields = '__all__'
        read_only_fields = ['treatment_id', 'created_at', 'updated_at']


class AppointmentSerializer(serializers.ModelSerializer):
    """Serializer for Appointment model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)

    class Meta:
        model = Appointment
        fields = '__all__'
        read_only_fields = ['appointment_id', 'created_at', 'updated_at']


class RiskAssessmentSerializer(serializers.ModelSerializer):
    """Serializer for Risk Assessment model"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    assessed_by_name = serializers.CharField(source='assessed_by.username', read_only=True)

    class Meta:
        model = RiskAssessment
        fields = '__all__'
        read_only_fields = ['assessment_id', 'created_at']
