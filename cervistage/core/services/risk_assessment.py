"""
Risk Assessment Service for Cervical Cancer Risk Calculation
Implements evidence-based risk scoring system based on established guidelines
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class RiskAssessmentService:
    """Service for calculating cervical cancer risk scores and recommendations"""

    # Risk factor weights based on clinical evidence
    RISK_WEIGHTS = {
        'age': 15,           # Maximum 15 points
        'hpv': 25,           # Maximum 25 points (highest weight)
        'smoking': 10,       # Maximum 10 points
        'alcohol': 5,        # Maximum 5 points
        'family_history': 15, # Maximum 15 points
        'hiv': 10,          # Maximum 10 points
        'diabetes': 5,       # Maximum 5 points
        'pregnancy': 5,      # Maximum 5 points (based on age at first pregnancy)
        'contraceptive': 5,  # Maximum 5 points
        'sexual_history': 5, # Maximum 5 points
    }

    # Risk thresholds
    RISK_THRESHOLDS = {
        'low': 30,          # 0-30: Low Risk
        'moderate': 60,      # 31-60: Moderate Risk
        'high': 80,         # 61-80: High Risk
        'very_high': 100    # 81-100: Very High Risk
    }

    @classmethod
    def calculate_risk_score(cls, assessment_data: Dict, patient_age: int) -> Tuple[int, str, Dict]:
        """
        Calculate comprehensive risk score (0-100) based on risk factors

        Args:
            assessment_data: Dictionary containing all risk factors
            patient_age: Patient's age in years

        Returns:
            Tuple of (risk_score, risk_level, risk_factors_breakdown)
        """
        risk_score = 0
        risk_factors = {}
        breakdown = {}

        # 1. Age Risk (0-15 points)
        age_score, age_details = cls._calculate_age_risk(patient_age)
        risk_score += age_score
        breakdown['age'] = age_details

        # 2. HPV Status Risk (0-25 points) - Highest weight
        hpv_score, hpv_details = cls._calculate_hpv_risk(assessment_data.get('hpv_risk'))
        risk_score += hpv_score
        breakdown['hpv'] = hpv_details

        # 3. Smoking Risk (0-10 points)
        smoking_score, smoking_details = cls._calculate_smoking_risk(
            assessment_data.get('smoking_status'),
            assessment_data.get('smoking_years')
        )
        risk_score += smoking_score
        breakdown['smoking'] = smoking_details

        # 4. Alcohol Use Risk (0-5 points)
        alcohol_score, alcohol_details = cls._calculate_alcohol_risk(assessment_data.get('alcohol_use'))
        risk_score += alcohol_score
        breakdown['alcohol'] = alcohol_details

        # 5. Family History Risk (0-15 points)
        family_score, family_details = cls._calculate_family_history_risk(
            assessment_data.get('family_history_risk'),
            assessment_data.get('family_history_details')
        )
        risk_score += family_score
        breakdown['family_history'] = family_details

        # 6. HIV Status Risk (0-10 points)
        hiv_score, hiv_details = cls._calculate_hiv_risk(assessment_data.get('hiv_status'))
        risk_score += hiv_score
        breakdown['hiv'] = hiv_details

        # 7. Diabetes Risk (0-5 points)
        diabetes_score, diabetes_details = cls._calculate_diabetes_risk(
            assessment_data.get('diabetes'),
            assessment_data.get('diabetes_type')
        )
        risk_score += diabetes_score
        breakdown['diabetes'] = diabetes_details

        # 8. Pregnancy History Risk (0-5 points)
        pregnancy_score, pregnancy_details = cls._calculate_pregnancy_risk(
            assessment_data.get('age_at_first_pregnancy'),
            assessment_data.get('number_of_pregnancies')
        )
        risk_score += pregnancy_score
        breakdown['pregnancy'] = pregnancy_details

        # 9. Oral Contraceptive Risk (0-5 points)
        contraceptive_score, contraceptive_details = cls._calculate_contraceptive_risk(
            assessment_data.get('oral_contraceptive_use'),
            assessment_data.get('oral_contraceptive_years')
        )
        risk_score += contraceptive_score
        breakdown['contraceptive'] = contraceptive_details

        # 10. Sexual History Risk (0-5 points)
        sexual_score, sexual_details = cls._calculate_sexual_history_risk(
            assessment_data.get('sexual_partners_count'),
            assessment_data.get('age_at_first_sexual_intercourse')
        )
        risk_score += sexual_score
        breakdown['sexual_history'] = sexual_details

        # 11. Additional Risk Factors
        additional_score, additional_details = cls._calculate_additional_risks(assessment_data)
        risk_score += additional_score
        breakdown['additional'] = additional_details

        # Ensure score doesn't exceed 100
        risk_score = min(risk_score, 100)

        # Determine risk level
        if risk_score <= cls.RISK_THRESHOLDS['low']:
            risk_level = 'low'
        elif risk_score <= cls.RISK_THRESHOLDS['moderate']:
            risk_level = 'moderate'
        elif risk_score <= cls.RISK_THRESHOLDS['high']:
            risk_level = 'high'
        else:
            risk_level = 'very_high'

        return risk_score, risk_level, breakdown

    @classmethod
    def _calculate_age_risk(cls, age: int) -> Tuple[int, Dict]:
        """Calculate age-related risk (0-15 points)"""
        if age < 21:
            return 0, {'score': 0, 'reason': 'Low risk age group'}
        elif 21 <= age <= 29:
            return 5, {'score': 5, 'reason': 'Early reproductive age'}
        elif 30 <= age <= 39:
            return 8, {'score': 8, 'reason': 'Moderate risk age group'}
        elif 40 <= age <= 49:
            return 12, {'score': 12, 'reason': 'Higher risk age group'}
        elif 50 <= age <= 59:
            return 15, {'score': 15, 'reason': 'Peak risk age group'}
        else:  # 60+
            return 10, {'score': 10, 'reason': 'Advanced age (lower screening rates)'})

    @classmethod
    def _calculate_hpv_risk(cls, hpv_status: str) -> Tuple[int, Dict]:
        """Calculate HPV-related risk (0-25 points) - Highest weight"""
        if hpv_status == 'positive_high_risk':
            return 25, {'score': 25, 'reason': 'High-risk HPV positive (highest risk factor)'}
        elif hpv_status == 'positive_low_risk':
            return 15, {'score': 15, 'reason': 'Low-risk HPV positive'}
        elif hpv_status == 'unknown':
            return 5, {'score': 5, 'reason': 'HPV status unknown (uncertain risk)'}
        else:  # negative
            return 0, {'score': 0, 'reason': 'HPV negative (baseline risk)'}

    @classmethod
    def _calculate_smoking_risk(cls, smoking_status: str, smoking_years: int = None) -> Tuple[int, Dict]:
        """Calculate smoking-related risk (0-10 points)"""
        if smoking_status == 'current_smoker':
            base_score = 10
            if smoking_years and smoking_years > 10:
                return 10, {'score': 10, 'reason': f'Current smoker ({smoking_years}+ years)'}
            return 8, {'score': 8, 'reason': 'Current smoker'}
        elif smoking_status == 'former_smoker':
            if smoking_years and smoking_years > 15:
                return 5, {'score': 5, 'reason': f'Former smoker (long-term history: {smoking_years} years)'}
            return 3, {'score': 3, 'reason': 'Former smoker (reduced risk)'}
        else:  # non_smoker
            return 0, {'score': 0, 'reason': 'Non-smoker (no additional risk)'}

    @classmethod
    def _calculate_alcohol_risk(cls, alcohol_use: str) -> Tuple[int, Dict]:
        """Calculate alcohol-related risk (0-5 points)"""
        if alcohol_use == 'heavy':
            return 5, {'score': 5, 'reason': 'Heavy alcohol use'}
        elif alcohol_use == 'moderate':
            return 3, {'score': 3, 'reason': 'Moderate alcohol use'}
        elif alcohol_use == 'occasional':
            return 1, {'score': 1, 'reason': 'Occasional alcohol use'}
        else:  # none
            return 0, {'score': 0, 'reason': 'No alcohol use'}

    @classmethod
    def _calculate_family_history_risk(cls, family_history: bool, details: str = None) -> Tuple[int, Dict]:
        """Calculate family history-related risk (0-15 points)"""
        if not family_history:
            return 0, {'score': 0, 'reason': 'No family history'}

        # Check for first-degree relative
        if details and ('mother' in details.lower() or 'sister' in details.lower()):
            return 15, {'score': 15, 'reason': 'First-degree relative with cervical cancer'}
        else:
            return 8, {'score': 8, 'reason': 'Family history of cervical cancer'}

    @classmethod
    def _calculate_hiv_risk(cls, hiv_status: str) -> Tuple[int, Dict]:
        """Calculate HIV-related risk (0-10 points)"""
        if hiv_status == 'positive':
            return 10, {'score': 10, 'reason': 'HIV positive (significantly elevated risk)'}
        elif hiv_status == 'unknown':
            return 3, {'score': 3, 'reason': 'HIV status unknown'}
        else:  # negative
            return 0, {'score': 0, 'reason': 'HIV negative'}

    @classmethod
    def _calculate_diabetes_risk(cls, diabetes: bool, diabetes_type: str = None) -> Tuple[int, Dict]:
        """Calculate diabetes-related risk (0-5 points)"""
        if not diabetes:
            return 0, {'score': 0, 'reason': 'No diabetes'}

        if diabetes_type == 'type2':
            return 5, {'score': 5, 'reason': 'Type 2 diabetes (elevated risk)'}
        else:
            return 3, {'score': 3, 'reason': 'Diabetes (potential risk factor)'}

    @classmethod
    def _calculate_pregnancy_risk(cls, age_at_first_pregnancy: int = None,
                                   number_of_pregnancies: int = None) -> Tuple[int, Dict]:
        """Calculate pregnancy-related risk (0-5 points)"""
        if age_at_first_pregnancy and age_at_first_pregnancy < 18:
            return 5, {'score': 5, 'reason': f'Early first pregnancy ({age_at_first_pregnancy} years)'}
        elif age_at_first_pregnancy and age_at_first_pregnancy < 20:
            return 3, {'score': 3, 'reason': f'Early first pregnancy ({age_at_first_pregnancy} years)'}
        elif number_of_pregnancies and number_of_pregnancies > 5:
            return 2, {'score': 2, 'reason': f'High parity ({number_of_pregnancies} pregnancies)'}
        else:
            return 0, {'score': 0, 'reason': 'Normal pregnancy history'}

    @classmethod
    def _calculate_contraceptive_risk(cls, contraceptive_use: str = None,
                                       years: int = None) -> Tuple[int, Dict]:
        """Calculate oral contraceptive-related risk (0-5 points)"""
        if contraceptive_use == 'current':
            if years and years > 5:
                return 5, {'score': 5, 'reason': f'Long-term OC use ({years}+ years)'}
            return 3, {'score': 3, 'reason': 'Current OC user'}
        elif contraceptive_use == 'past':
            if years and years > 10:
                return 2, {'score': 2, 'reason': f'Long-term OC history ({years} years)'}
            return 1, {'score': 1, 'reason': 'Past OC use (reduced risk)'}
        else:
            return 0, {'score': 0, 'reason': 'No OC use'}

    @classmethod
    def _calculate_sexual_history_risk(cls, partner_count: int = None,
                                         age_first_intercourse: int = None) -> Tuple[int, Dict]:
        """Calculate sexual history-related risk (0-5 points)"""
        score = 0
        reasons = []

        if partner_count and partner_count > 5:
            score += 3
            reasons.append(f'Multiple sexual partners ({partner_count})')

        if age_first_intercourse and age_first_intercourse < 16:
            score += 3
            reasons.append(f'Early sexual debut ({age_first_intercourse} years)')

        if score == 0:
            return 0, {'score': 0, 'reason': 'Normal sexual history'}
        return min(score, 5), {'score': min(score, 5), 'reason': ', '.join(reasons)}

    @classmethod
    def _calculate_additional_risks(cls, data: Dict) -> Tuple[int, Dict]:
        """Calculate additional risk factors"""
        score = 0
        factors = []

        if data.get('immunocompromised'):
            score += 10
            factors.append('Immunocompromised')

        if data.get('previous_abnormal_pap'):
            score += 8
            factors.append('Previous abnormal Pap smear')

        if not data.get('hpv_vaccinated'):
            score += 5
            factors.append('Unvaccinated against HPV')

        if data.get('symptoms_risk') == 'symptomatic_concerning':
            score += 5
            factors.append('Concerning symptoms')

        if score == 0:
            return 0, {'score': 0, 'reason': 'No additional risk factors'}
        return score, {'score': score, 'reason': ', '.join(factors)}

    @classmethod
    def generate_recommendations(cls, risk_score: int, risk_level: str,
                                  patient_age: int, has_ai_analysis: bool = False) -> str:
        """
        Generate screening and clinical recommendations based on risk assessment

        Args:
            risk_score: Calculated risk score (0-100)
            risk_level: Risk level category
            patient_age: Patient's age
            has_ai_analysis: Whether AI analysis was performed

        Returns:
            Recommendation text
        """
        recommendations = []

        # Screening frequency recommendation
        if risk_level == 'low':
            if patient_age < 21:
                recommendations.append("Screening: Begin screening at age 21")
            elif patient_age < 30:
                recommendations.append("Screening: Pap smear every 3 years")
            else:
                recommendations.append("Screening: Pap smear every 3 years or Pap + HPV test every 5 years")
        elif risk_level == 'moderate':
            recommendations.append("Screening: Annual Pap smear recommended")
            recommendations.append("Consider: HPV co-testing with Pap smear")
        elif risk_level == 'high':
            recommendations.append("Screening: Semi-annual (every 6 months) Pap smear required")
            recommendations.append("Include: HPV DNA testing with each screening")
            recommendations.append("Consider: Colposcopy referral for comprehensive evaluation")
        else:  # very_high
            recommendations.append("Screening: Quarterly (every 3 months) follow-up required")
            recommendations.append("Urgent: Immediate colposcopy referral recommended")
            recommendations.append("Consider: biopsy for definitive diagnosis")

        # HPV Vaccination recommendation
        if patient_age < 26:
            recommendations.append("Vaccination: HPV vaccination strongly recommended if not already vaccinated")
        elif patient_age < 45:
            recommendations.append("Vaccination: Discuss HPV vaccination benefits with healthcare provider")

        # Lifestyle modifications
        if risk_level in ['high', 'very_high']:
            recommendations.append("Lifestyle: Smoking cessation program recommended")
            recommendations.append("Lifestyle: Limit alcohol consumption")
            recommendations.append("Lifestyle: Maintain healthy diet and regular exercise")

        # Additional precautions
        if risk_level == 'very_high':
            recommendations.append("Precautions: Safe sexual practices - consistent condom use")
            recommendations.append("Monitoring: Regular gynecological check-ups every 3 months")
            recommendations.append("Education: Patient education on symptom awareness")

        # AI Analysis recommendation
        if has_ai_analysis:
            recommendations.append("AI Analysis: Results indicate need for clinical correlation and follow-up")
        else:
            recommendations.append("AI Analysis: Consider AI-assisted screening for comprehensive evaluation")

        # Add urgency indicator
        if risk_level == 'very_high':
            recommendations.insert(0, "⚠️ URGENT: High-risk profile requiring immediate clinical attention")
        elif risk_level == 'high':
            recommendations.insert(0, "⚠️ ATTENTION: Elevated risk profile - ensure timely follow-up")

        return "\n".join(recommendations)

    @classmethod
    def determine_screening_frequency(cls, risk_level: str) -> str:
        """Determine appropriate screening frequency based on risk level"""
        frequency_map = {
            'low': 'routine',
            'moderate': 'annual',
            'high': 'semi_annual',
            'very_high': 'quarterly'
        }
        return frequency_map.get(risk_level, 'routine')

    @classmethod
    def calculate_next_review_date(cls, risk_level: str) -> datetime.date:
        """Calculate recommended date for next risk assessment review"""
        from django.utils import timezone

        review_intervals = {
            'low': timedelta(days=365),      # 1 year
            'moderate': timedelta(days=180),  # 6 months
            'high': timedelta(days=90),      # 3 months
            'very_high': timedelta(days=30)  # 1 month
        }

        interval = review_intervals.get(risk_level, timedelta(days=365))
        return (timezone.now().date() + interval)
