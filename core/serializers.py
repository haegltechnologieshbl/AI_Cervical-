from rest_framework import serializers
from .models import Analysis


class AnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = "__all__"
        read_only_fields = [
            "analysis_id", "predicted_class", "predicted_label",
            "probabilities", "confidence", "uncertainty",
            "confidence_level", "recommendation", "created_at",
        ]


class AnalyzeRequestSerializer(serializers.Serializer):
    file = serializers.ListField(child=serializers.ImageField(), required=False)
    
    def validate(self, data):
        # Support single or multiple file uploads
        if 'file' not in data or not data['file']:
            raise serializers.ValidationError("At least one image file is required.")
        return data
