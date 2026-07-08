import re
from rest_framework import serializers
from .models import (
    Analysis, Patient,
    PapSmearTest, CheckupRecommendation
)


class PatientSerializer(serializers.ModelSerializer):
    """Comprehensive serializer for Patient model with all fields"""
    full_name = serializers.SerializerMethodField()
    age_calculated = serializers.SerializerMethodField()
    analyses_count = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = [
            'patient_id', 'first_name', 'last_name', 'full_name',
            'date_of_birth', 'age', 'age_calculated', 'gender',
            'phone', 'email', 'address',
            'hpv_status', 'last_screening_date', 'pregnancy_status',
            'medical_history', 'medications', 'allergies', 'family_history',
            'notes', 'created_at', 'updated_at', 'analyses_count', 'created_by',
            'assigned_doctor', 'assigned_technician',
        ]
        read_only_fields = ['patient_id', 'created_at', 'updated_at', 'created_by']

    def get_full_name(self, obj):
        return obj.full_name

    def get_age_calculated(self, obj):
        return obj.age_calculated

    def get_analyses_count(self, obj):
        return obj.analyses.count()


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


class PapSmearTestSerializer(serializers.ModelSerializer):
    """Serializer for Pap Smear Test"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    technician_name = serializers.CharField(source='technician.username', read_only=True)

    class Meta:
        model = PapSmearTest
        fields = '__all__'
        read_only_fields = ['test_id', 'test_date', 'technician']

class CheckupRecommendationSerializer(serializers.ModelSerializer):
    """Serializer for Doctor's Checkup Recommendation"""
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    doctor_name = serializers.CharField(source='doctor.username', read_only=True)

    class Meta:
        model = CheckupRecommendation
        fields = '__all__'
        read_only_fields = ['checkup_id', 'created_at', 'doctor']
