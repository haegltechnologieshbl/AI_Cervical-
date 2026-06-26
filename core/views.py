from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from datetime import datetime, timedelta
from django.utils import timezone
from django.contrib.auth import get_user_model, authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .models import (
    Analysis, Patient,
    ClinicalNote, PapSmearTest, CheckupRecommendation, Notification,
)
from .serializers import (
    AnalysisSerializer, AnalyzeRequestSerializer, PatientSerializer, PatientListSerializer,
)
from services.inference_vps import VPSInferenceService
from core.services import ImageValidator
import cv2
import numpy as np
import os
import uuid
from django.conf import settings
from PIL import Image
from django.core.mail import send_mail
from django.core.cache import cache
import random
import string

User = get_user_model()

RECOMMENDED_CHECKUP_OPTIONS = [
    "HPV PCR Testing",
    "Colposcopy",
    "Histopathology (Gold Standard)",
    "MRI only for confirmed invasive cancer",
    "Treatment & Follow-up",
]


def get_selected_checkups(request):
    selected = request.data.getlist('recommended_checkups')
    return [item for item in selected if item in RECOMMENDED_CHECKUP_OPTIONS]


def get_patient_queryset_for_user(user):
    """Return patients the current user is allowed to work with."""
    user_role = user.profile.role if hasattr(user, 'profile') else 'user'
    if user_role == 'admin' or user.is_superuser:
        return Patient.objects.all()
    if user_role == 'senior_technician':
        return Patient.objects.filter(
            assigned_technician=user,
            assigned_doctor__isnull=False,
        )
    if user_role == 'user':
        return Patient.objects.filter(assigned_doctor=user)
    return Patient.objects.none()


def get_allowed_patient_for_user(user, patient_id):
    """Look up a patient only within the user's permitted patient set."""
    if not patient_id:
        user_role = user.profile.role if hasattr(user, 'profile') else 'user'
        if user_role == 'senior_technician':
            return None, Response(
                {"error": "Please select a patient assigned to you before running analysis."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return None, None
    try:
        return get_patient_queryset_for_user(user).get(patient_id=patient_id), None
    except Patient.DoesNotExist:
        return None, Response(
            {"error": "Patient not found or not assigned to you."},
            status=status.HTTP_403_FORBIDDEN,
        )

# Safe print function for Windows console compatibility
def safe_print(*args, **kwargs):
    """Print function that handles Unicode characters safely on Windows."""
    import sys
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode with replacement for unsupported characters
        encoded_args = [str(arg).encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8') for arg in args]
        print(*encoded_args, **kwargs)



# ─── Role-based access control ───────────────────────────────────────────────

class RoleRequiredMixin(LoginRequiredMixin):
    """Restrict a class-based view to specific profile roles.

    Set ``allowed_roles`` on the view (e.g. ``allowed_roles = ('admin',)``).
    Superusers are always treated as ``'admin'``. Unauthenticated users are
    redirected to login; authenticated users with the wrong role are sent home.
    """
    allowed_roles = ()

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return super().dispatch(request, *args, **kwargs)  # -> login redirect
        role = getattr(getattr(request.user, 'profile', None), 'role', None)
        if request.user.is_superuser:
            role = 'admin'
        if self.allowed_roles and role not in self.allowed_roles:
            messages.error(request, "Access denied. You don't have permission to view this page.")
            return redirect('home')
        return super().dispatch(request, *args, **kwargs)


class CanAnalyze(IsAuthenticated):
    """Authenticated users except doctors may run AI analysis.

    Per the clinical workflow, doctors do not analyse data themselves — the
    technician runs the Pap smear test/analysis. Doctors (role ``'user'``) are
    therefore denied access to the analysis endpoints.
    """
    message = "Doctors are not permitted to run analysis. Assign the patient to a technician for Pap smear testing."

    def has_permission(self, request, view):
        if not super().has_permission(request, view):
            return False
        role = getattr(getattr(request.user, 'profile', None), 'role', None)
        return role != 'user'


# ─── Template Views ────────────────────────────────────────────────────────────

class HomePageView(View):
    """GET / - Landing page (no authentication required)"""
    def get(self, request):
        # If authenticated, send the user to the dashboard router, which forwards
        # each role to its correct workspace (admin / doctor / technician / patient).
        if request.user.is_authenticated:
            return redirect('/dashboard/')

        # Show landing page for non-authenticated users
        return render(request, "home.html")


class AnalyzePageView(View):
    """GET /analyze/ - AI analysis tool (requires authentication)"""
    def get(self, request):
        # Require authentication for analyze
        if not request.user.is_authenticated:
            next_url = request.get_full_path()
            return redirect(f'/login/?next={next_url}')

        user_role = request.user.profile.role if hasattr(request.user, 'profile') else None

        # Doctors do not analyse data themselves — the technician runs the test.
        if user_role == 'user':
            messages.info(request, "Doctors don't run analysis. Assign the patient to a technician for Pap smear testing.")
            return redirect('/doctor/dashboard/')

        if user_role == 'senior_technician':
            if request.path == '/analyze/':
                query_string = f'?{request.META.get("QUERY_STRING")}' if request.META.get("QUERY_STRING") else ''
                return redirect(f'/technician/analyze/{query_string}')
            # Server-side patient list (session-authenticated) so the dropdown
            # never depends on a possibly-stale JWT in localStorage.
            import json as _json
            patients = get_patient_queryset_for_user(request.user).order_by('first_name', 'last_name')
            patients_json = _json.dumps([
                {
                    'patient_id': str(p.patient_id),
                    'full_name': p.full_name,
                    'age': p.age if p.age is not None else '',
                    'hpv_status': p.hpv_status,
                }
                for p in patients
            ])
            return render(request, "technician/technician_analyze.html", {
                'assigned_patients': patients,
                'patients_json': patients_json,
            })

        return render(request, "analyze.html")


class DemoDatasetView(View):
    """GET /api/demo/<category>/ - List demo images for a dataset category"""
    ALLOWED = {'hsil', 'lsil', 'moderate_dysplasia', 'carcinoma'}
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def get(self, request, category):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Authentication required'}, status=401)

        if category not in self.ALLOWED:
            return JsonResponse({'error': 'Invalid category'}, status=400)

        import os
        from django.conf import settings

        folder = os.path.join(settings.MEDIA_ROOT, 'demo', category)
        images = []
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.IMAGE_EXTS:
                    images.append({
                        'name': fname,
                        'url': f'{settings.MEDIA_URL}demo/{category}/{fname}',
                    })

        return JsonResponse({'category': category, 'images': images})



class HistoryPageView(View):
    """GET /history/ - Analysis history with pagination and filters"""
    def get(self, request):
        # Require authentication for history
        if not request.user.is_authenticated:
            return redirect('/login/?next=/history/')

        from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
        from django.db.models import Q
        import datetime

        # Get filter parameters
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')
        patient_id = request.GET.get('patient_id')
        result_filter = request.GET.get('result')
        search_query = request.GET.get('search', '').strip()
        search_type = request.GET.get('search_type', 'all')

        # Base queryset with role filtering
        user_role = request.user.profile.role
        _sel = ("created_by", "created_by__profile", "patient")
        if user_role == 'admin' or request.user.is_superuser:
            analyses_qs = Analysis.objects.select_related(*_sel)
        elif user_role == 'user':
            # Doctor — analyses belonging to patients assigned to this doctor
            analyses_qs = Analysis.objects.filter(
                patient__assigned_doctor=request.user
            ).select_related(*_sel)
        elif user_role == 'patient':
            # Patient — only their own analyses
            analyses_qs = Analysis.objects.filter(
                patient__user_account=request.user
            ).select_related(*_sel)
        else:
            # Technician — analyses they performed
            analyses_qs = Analysis.objects.filter(
                created_by=request.user
            ).select_related(*_sel)

        # Filter out non-primary batch analyses (show only the primary analysis from each batch)
        # This prevents showing 10 separate entries when user uploads 10 images as a batch
        analyses_qs = analyses_qs.filter(
            Q(batch_id__isnull=True) | Q(is_batch_primary=True)
        )

        # Apply filters
        if date_from:
            try:
                date_from_obj = datetime.datetime.strptime(date_from, '%Y-%m-%d').date()
                analyses_qs = analyses_qs.filter(created_at__date__gte=date_from_obj)
            except ValueError:
                pass

        if date_to:
            try:
                date_to_obj = datetime.datetime.strptime(date_to, '%Y-%m-%d').date()
                analyses_qs = analyses_qs.filter(created_at__date__lte=date_to_obj)
            except ValueError:
                pass

        if patient_id:
            analyses_qs = analyses_qs.filter(patient_id=patient_id)

        if result_filter:
            try:
                result_int = int(result_filter)
                analyses_qs = analyses_qs.filter(predicted_class=result_int)
            except ValueError:
                pass

        if search_query:
            analyses_qs = analyses_qs.filter(
                Q(analysis_id__icontains=search_query) |
                Q(predicted_label__icontains=search_query) |
                Q(patient__full_name__icontains=search_query)
            )

        # Order by date descending
        analyses_qs = analyses_qs.order_by("-created_at")

        # Pagination
        items_per_page = 20
        page = request.GET.get('page', 1)
        paginator = Paginator(analyses_qs, items_per_page)

        try:
            analyses_page = paginator.page(page)
        except PageNotAnInteger:
            analyses_page = paginator.page(1)
        except EmptyPage:
            analyses_page = paginator.page(paginator.num_pages)

        # Attach each patient's latest doctor checkup recommendation to the rows
        patient_ids = {a.patient_id for a in analyses_page if a.patient_id}
        if patient_ids:
            from .models import CheckupRecommendation
            latest_checkups = {}
            for rec in CheckupRecommendation.objects.filter(
                patient_id__in=patient_ids
            ).order_by('patient_id', '-created_at'):
                latest_checkups.setdefault(rec.patient_id, rec)
            for a in analyses_page:
                rec = latest_checkups.get(a.patient_id)
                a.doctor_checkups = rec.recommended_list() if rec else []
        else:
            for a in analyses_page:
                a.doctor_checkups = []

        # Get all patients for filter dropdown (admin only)
        patients_list = []
        if user_role == 'admin':
            from .models import Patient
            patients_list = Patient.objects.only('patient_id', 'full_name').order_by('full_name')[:100]

        # Preserve filter parameters for pagination
        filter_params = {}
        if date_from:
            filter_params['date_from'] = date_from
        if date_to:
            filter_params['date_to'] = date_to
        if patient_id:
            filter_params['patient'] = patient_id
        if result_filter:
            filter_params['result'] = result_filter
        if search_query:
            filter_params['search'] = search_query

        context = {
            "analyses": analyses_page,
            "user_role": user_role,
            "patients": patients_list,
            "filter_params": filter_params,
            "total_count": paginator.count,
            "page_obj": analyses_page,
        }

        # Choose role-specific template
        if user_role == 'user':
            template_name = "doctor/doctor_history.html"
        elif user_role == 'senior_technician':
            template_name = "technician/technician_history.html"
        else:
            template_name = "history.html"

        return render(request, template_name, context)


class DeleteAnalysisView(View):
    """POST /analysis/<id>/delete/ - delete a single analysis.

    Allowed for the analysis owner (the technician who ran it) or an admin.
    Redirects back to the page the request came from.
    """
    def post(self, request, analysis_id):
        if not request.user.is_authenticated:
            return redirect('/login/')
        analysis = get_object_or_404(Analysis, pk=analysis_id)
        role = request.user.profile.role if hasattr(request.user, 'profile') else None
        is_assigned_doctor = (
            role == 'user' and analysis.patient_id is not None
            and analysis.patient.assigned_doctor_id == request.user.id
        )
        if (request.user.is_superuser or role == 'admin'
                or analysis.created_by_id == request.user.id
                or is_assigned_doctor):
            # Delete the whole batch when removing a batch primary
            if analysis.batch_id:
                Analysis.objects.filter(batch_id=analysis.batch_id).delete()
            else:
                analysis.delete()
            messages.success(request, "Analysis deleted.")
        else:
            messages.error(request, "You don't have permission to delete this analysis.")
        return redirect(request.META.get('HTTP_REFERER') or 'technician-history')


class LoginPageView(View):
    """GET /login/ - Login page"""
    def get(self, request):
        return render(request, "login.html")


def build_dashboard_context(user):
    """Analytics context shared by the doctor dashboard and the generic fallback."""
    from django.db.models import Count, Q
    from django.db.models.functions import TruncDate
    from datetime import timedelta
    import json

    total_analyses = Analysis.objects.filter(created_by=user).count()

    seven_days_ago = timezone.now() - timedelta(days=7)
    recent_analyses_count = Analysis.objects.filter(
        created_by=user, created_at__gte=seven_days_ago
    ).count()

    disease_distribution = list(Analysis.objects.filter(created_by=user).values('predicted_class', 'predicted_label').annotate(
        count=Count('analysis_id')
    ).order_by('predicted_class'))

    thirty_days_ago = timezone.now() - timedelta(days=30)
    date_trends = list(Analysis.objects.filter(
        created_by=user, created_at__gte=thirty_days_ago
    ).annotate(date=TruncDate('created_at')).values('date').annotate(
        count=Count('analysis_id')
    ).order_by('date'))

    risk_distribution = []
    for risk in ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']:
        count = Analysis.objects.filter(
            created_by=user,
            predicted_label__icontains=risk.split()[0] if risk != 'Very High Risk' else 'Carcinoma'
        ).count()
        risk_distribution.append({'level': risk, 'count': count})

    confidence_distribution = list(Analysis.objects.filter(created_by=user).values('confidence_level').annotate(
        count=Count('analysis_id')
    ).order_by('-count'))

    total_patients_tested = Analysis.objects.filter(
        created_by=user, patient__isnull=False
    ).values('patient').distinct().count()
    patient_analyses = Analysis.objects.filter(created_by=user, patient__isnull=False).count()

    recent_activity = Analysis.objects.filter(created_by=user).filter(
        Q(batch_id__isnull=True) | Q(is_batch_primary=True)
    ).select_related('patient').order_by('-created_at')[:10]

    for item in disease_distribution:
        item['percentage'] = round((item['count'] / total_analyses) * 100, 1) if total_analyses > 0 else 0

    for item in date_trends:
        if item.get('date'):
            item['date'] = item['date'].strftime('%Y-%m-%d')

    daily_users_tested = list(Analysis.objects.filter(
        created_by=user, created_at__gte=thirty_days_ago, patient__isnull=False
    ).annotate(date=TruncDate('created_at')).values('date').annotate(
        unique_users=Count('patient', distinct=True)
    ).order_by('date'))
    for item in daily_users_tested:
        if item.get('date'):
            item['date'] = item['date'].strftime('%Y-%m-%d')

    # --- Doctor-specific context ---
    total_patients = Patient.objects.filter(assigned_doctor=user).count()
    pending_reviews = 0
    pending_pap_smears = []
    high_risk_count = 0
    monthly_analyses = 0

    if getattr(getattr(user, 'profile', None), 'role', None) == 'user':
        assigned_patients = Patient.objects.filter(assigned_doctor=user)
        pap_smears_without_checkups = PapSmearTest.objects.filter(
            patient__in=assigned_patients
        ).exclude(associated_checkups__doctor=user).select_related('patient', 'technician').order_by('-test_date')
        pending_reviews = pap_smears_without_checkups.count()
        pending_pap_smears = list(pap_smears_without_checkups[:5])

        high_risk_count = Analysis.objects.filter(
            patient__in=assigned_patients, predicted_class__in=[3, 4]  # HSIL=3, Carcinoma=4
        ).values('patient').distinct().count()

        month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_analyses = Analysis.objects.filter(
            created_by=user, created_at__gte=month_start
        ).count()

    return {
        'total_analyses': total_analyses,
        'recent_analyses_count': recent_analyses_count,
        'disease_distribution_json': json.dumps(disease_distribution),
        'date_trends_json': json.dumps(date_trends),
        'risk_distribution_json': json.dumps(risk_distribution),
        'confidence_distribution_json': json.dumps(confidence_distribution),
        'daily_users_tested_json': json.dumps(daily_users_tested),
        'total_patients': total_patients,
        'total_patients_tested': total_patients_tested,
        'patient_analyses': patient_analyses,
        'recent_activity': recent_activity,
        'pending_reviews': pending_reviews,
        'pending_pap_smears': pending_pap_smears,
        'high_risk_count': high_risk_count,
        'monthly_analyses': monthly_analyses,
    }


def build_doctor_dashboard_context(user):
    """Context for the doctor dashboard — only the doctor's own workload.

    Doctors don't run analysis, so this deliberately omits all AI analytics and
    surfaces just: assigned patients, Pap smear results awaiting review,
    high-risk patients, and checkups the doctor has completed.
    """
    assigned_patients = Patient.objects.filter(assigned_doctor=user)
    total_patients = assigned_patients.count()

    pap_smears_without_checkups = PapSmearTest.objects.filter(
        patient__in=assigned_patients
    ).exclude(associated_checkups__doctor=user).select_related('patient', 'technician').order_by('-test_date')
    pending_reviews = pap_smears_without_checkups.count()
    pending_pap_smears = list(pap_smears_without_checkups[:8])

    high_risk_count = Analysis.objects.filter(
        patient__in=assigned_patients, predicted_class__in=[3, 4]  # HSIL=3, Carcinoma=4
    ).values('patient').distinct().count()

    completed_reviews = CheckupRecommendation.objects.filter(
        doctor=user
    ).select_related('patient', 'pap_smear_test').order_by('-created_at')

    return {
        'total_patients': total_patients,
        'pending_reviews': pending_reviews,
        'pending_pap_smears': pending_pap_smears,
        'high_risk_count': high_risk_count,
        'completed_reviews_count': completed_reviews.count(),
        'recent_reviews': list(completed_reviews[:10]),
    }


class DashboardPageView(View):
    """GET /dashboard/ - Routes each role to its own dashboard."""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/dashboard/')
        role = getattr(getattr(request.user, 'profile', None), 'role', None)
        if request.user.is_superuser or role == 'admin':
            return redirect('/admin/')
        if role == 'senior_technician':
            return redirect('/technician/dashboard/')
        if role == 'patient':
            return redirect('/patient/dashboard/')
        if role == 'user':  # 'user' is the Doctor role
            return redirect('/doctor/dashboard/')
        # Fallback for any authenticated user without a specialised role
        return render(request, "user_dashboard.html", build_dashboard_context(request.user))


class AnalysisDetailView(View):
    """GET /analysis/<analysis_id> - Detailed analysis results page"""
    def get(self, request, analysis_id):
        analysis = get_object_or_404(Analysis, pk=analysis_id)

        # Access control for authenticated staff — each role only sees analyses
        # within its scope. (Unauthenticated share-link viewers are unaffected.)
        if request.user.is_authenticated:
            user_role = request.user.profile.role if hasattr(request.user, 'profile') else None
            if not (request.user.is_superuser or user_role == 'admin'):
                p = analysis.patient
                if user_role == 'user':
                    allowed = bool(p) and p.assigned_doctor_id == request.user.id
                elif user_role == 'senior_technician':
                    allowed = analysis.created_by_id == request.user.id or (bool(p) and p.assigned_technician_id == request.user.id)
                elif user_role == 'patient':
                    allowed = bool(p) and p.user_account_id == request.user.id
                else:
                    allowed = analysis.created_by_id == request.user.id
                if not allowed:
                    messages.error(request, "You don't have access to this analysis.")
                    return redirect('/history/')

        # If this is part of a batch but not the primary analysis, redirect to primary
        if analysis.batch_id and not analysis.is_batch_primary:
            try:
                primary_analysis = Analysis.objects.filter(batch_id=analysis.batch_id, is_batch_primary=True).first()
                if primary_analysis:
                    return redirect(f'/analysis/{primary_analysis.analysis_id}')
            except:
                pass  # If primary not found, show current analysis

        # Convert probabilities and scores to percentages for display
        probabilities_pct = {k: round(float(v) * 100, 2) for k, v in analysis.probabilities.items()}
        confidence_pct = round(analysis.confidence * 100, 1)

        # Fetch batch information if this is a batch primary analysis
        batch_analyses = []
        batch_data = None
        if analysis.batch_id and analysis.is_batch_primary:
            batch_analyses = list(Analysis.objects.filter(batch_id=analysis.batch_id).order_by('batch_position'))
            batch_data = analysis.batch_aggregated_data
            print(f"[DEBUG] Loading batch: batch_id={analysis.batch_id}, batch_analyses count={len(batch_analyses)}")
            # Convert batch confidence to percentage for display
            if batch_data:
                # Handle both flat and nested batch_data structures
                if 'averaged_prediction' not in batch_data and 'confidence' in batch_data:
                    # Convert flat structure to nested for template compatibility
                    batch_data['averaged_prediction'] = {
                        'predicted_class': batch_data.get('predicted_class'),
                        'predicted_label': batch_data.get('predicted_label'),
                        'stage_label': batch_data.get('stage_label'),
                        'probabilities': batch_data.get('probabilities'),
                        'confidence': batch_data.get('confidence'),
                        'confidence_level': batch_data.get('confidence_level'),
                        'risk_level': batch_data.get('risk_level'),
                        'risk_color': batch_data.get('risk_color'),
                        'recommendation': batch_data.get('recommendation'),
                        'explanation': batch_data.get('explanation'),
                    }
                if 'averaged_prediction' in batch_data:
                    batch_data['averaged_prediction']['confidence_pct'] = round(
                        batch_data['averaged_prediction'].get('confidence', 0) * 100, 1
                    )
            # Add confidence percentage to each batch analysis for display
            for ba in batch_analyses:
                ba.confidence_pct = round(ba.confidence * 100, 1)
                # Compute probabilities percentage for each batch analysis
                ba.probabilities_pct = {k: round(v * 100, 1) for k, v in ba.probabilities.items()}
                print(f"[DEBUG] Batch analysis {ba.batch_position}: confidence={ba.confidence}, confidence_pct={ba.confidence_pct}")

        # Get all analyses for this patient to show thumbnail gallery
        patient_analyses = []
        if analysis.patient:
            patient_analyses = list(
                Analysis.objects.filter(
                    patient=analysis.patient
                ).order_by('-created_at').select_related('patient')
            )
            # Add confidence percentage to each patient analysis
            for pa in patient_analyses:
                pa.confidence_pct = round(pa.confidence * 100, 1)
                pa.is_current = (pa.analysis_id == analysis.analysis_id)

        # Technician's Pap smear report and the doctor's review for this patient
        pap_smear = None
        checkup = None
        doctor_checkups = []
        if analysis.patient:
            from .models import CheckupRecommendation, PapSmearTest
            pap_smear = PapSmearTest.objects.filter(
                patient=analysis.patient
            ).select_related('technician').order_by('-test_date').first()
            checkup = CheckupRecommendation.objects.filter(
                patient=analysis.patient
            ).select_related('doctor').order_by('-created_at').first()
            if checkup:
                doctor_checkups = checkup.recommended_list()

        context = {
            'analysis': analysis,
            'probabilities': probabilities_pct,
            'confidence_pct': confidence_pct,
            'batch_analyses': batch_analyses,
            'batch_data': batch_data,
            'is_batch': bool(analysis.batch_id),
            'patient_analyses': patient_analyses,
            'doctor_checkups': doctor_checkups,
            'pap_smear': pap_smear,
            'checkup': checkup,
        }
        role = request.user.profile.role if hasattr(request.user, 'profile') else None
        if role == 'user':
            template_name = "doctor/doctor_analysis_detail.html"
        elif role == 'senior_technician':
            template_name = "technician/technician_analysis_detail.html"
        elif role == 'patient':
            template_name = "patient/patient_analysis_detail.html"
        else:
            template_name = "analysis_detail.html"

        return render(request, template_name, context)



class UserProfilePageView(View):
    """GET /profile/ - User profile page"""
    def get(self, request):
        # Require authentication
        if not request.user.is_authenticated:
            return redirect('/login/?next=/profile/')

        user = request.user
        context = {
            'user': user,
        }
        # Choose role-specific template
        role = request.user.profile.role if hasattr(request.user, 'profile') else None
        if role == 'user':
            template_name = "doctor/doctor_profile.html"
        elif role == 'senior_technician':
            template_name = "technician/technician_profile.html"
        else:
            template_name = "user_profile.html"

        return render(request, template_name, context)


class StatsView(APIView):
    """GET /api/v1/stats - API endpoint for statistics"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        total_analyses = Analysis.objects.count()

        # Recent analyses count
        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_analyses = Analysis.objects.filter(
            created_at__gte=seven_days_ago
        ).count()

        # Severity distribution
        severity_dist = Analysis.objects.values('predicted_label').annotate(
            count=Count('analysis_id')
        ).order_by('-count')

        return Response({
            'total_analyses': total_analyses,
            'recent_analyses': recent_analyses,
            'severity_distribution': list(severity_dist),
        })


class HospitalDashboardStatsView(APIView):
    """GET /api/v1/dashboard/hospital-stats - Comprehensive hospital management dashboard statistics"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from datetime import timedelta

        today = timezone.now().date()
        month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0)

        # Safely determine user role (guard against missing profile)
        try:
            user_role = request.user.profile.role
        except Exception:
            user_role = 'user'
        
        is_admin = user_role == 'admin' or request.user.is_superuser

        # --- Querysets scoped by role ---
        if is_admin:
            # Admin sees ALL data across ALL users
            patient_queryset = Patient.objects.all()
            analysis_queryset = Analysis.objects.all()
        else:
            patient_queryset = Patient.objects.filter(created_by=request.user)
            analysis_queryset = Analysis.objects.filter(created_by=request.user)

        # Patient Statistics
        total_patients_registered = patient_queryset.count()

        # Count patients who have had at least one AI analysis
        total_patients_screened = patient_queryset.filter(
            analyses__isnull=False
        ).distinct().count()

        # High-risk patients: linked to an HSIL or Carcinoma analysis (class 3 or 4)
        high_risk_patients = patient_queryset.filter(
            analyses__predicted_class__gte=3
        ).distinct().count()

        recent_visits = patient_queryset.filter(created_at__date=today).count()

        # Analysis Statistics
        total_analyses = analysis_queryset.count()

        # Count unique patients for positive/negative cases (not total analyses)
        # Positive cases: patients whose most recent analysis is class 3 or 4 (HSIL/Carcinoma)
        positive_patients = patient_queryset.filter(
            analyses__predicted_class__gte=3
        ).distinct().count()

        # Negative cases: patients whose most recent analysis is class 0, 1, or 2 (Normal/ASC-US/LSIL)
        negative_patients = patient_queryset.filter(
            analyses__predicted_class__lte=2
        ).distinct().count()

        # For patients with multiple analyses, only count their latest result
        # Get the most recent analysis for each patient
        from django.db.models import Max

        latest_analyses = Analysis.objects.filter(
            patient__in=patient_queryset
        ).values('patient').annotate(
            latest_analysis=Max('analysis_id')
        )

        latest_analysis_ids = [item['latest_analysis'] for item in latest_analyses]

        # Recount based on latest analysis only
        positive_cases = Analysis.objects.filter(
            analysis_id__in=latest_analysis_ids,
            predicted_class__gte=3
        ).count()

        negative_cases = Analysis.objects.filter(
            analysis_id__in=latest_analysis_ids,
            predicted_class__lte=2
        ).count()

        pending_reviews = analysis_queryset.filter(
            recommendation__isnull=True
        ).count()
        analyses_today = analysis_queryset.filter(created_at__date=today).count()

        # Stage Distribution
        stage_dist = analysis_queryset.values('predicted_class').annotate(
            count=Count('analysis_id')
        ).order_by('predicted_class')

        stage_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for item in stage_dist:
            stage_distribution[item['predicted_class']] = item['count']

        # Monthly Detection Rate
        month_screened = analysis_queryset.filter(
            created_at__gte=month_start
        ).count()
        month_positive = analysis_queryset.filter(
            created_at__gte=month_start,
            predicted_class__gte=3
        ).count()

        monthly_detection_rate = (month_positive / month_screened * 100) if month_screened > 0 else 0

        # Replace this block in HospitalDashboardStatsView:


# AND ALSO REMOVE the second duplicate block with tzinfo

        # REPLACE WITH this single block:
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_patient_dates = patient_queryset.filter(
            created_at__gte=thirty_days_ago
        ).values_list('created_at', flat=True)

        registrations_by_day = {}
        for created_at in recent_patient_dates:
            if created_at is not None:
                key = timezone.localtime(created_at).date().strftime('%b %d')
                registrations_by_day[key] = registrations_by_day.get(key, 0) + 1

        from datetime import datetime as dt
        current_year = timezone.now().year
        try:
            registrations_by_day = dict(sorted(
                registrations_by_day.items(),
                key=lambda x: dt.strptime(f"{x[0]} {current_year}", '%b %d %Y')
            ))
        except Exception:
            pass

        # Calculate registrations in last 7 days
        seven_days_ago = timezone.now() - timedelta(days=7)
        registrations_week = patient_queryset.filter(
            created_at__gte=seven_days_ago
        ).count()

        return Response({
            'patient_stats': {
                'total_registered': total_patients_registered,
                'total_screened': total_patients_screened,
                'high_risk': high_risk_patients,
                'recent_visits': recent_visits,
                'registrations_week': registrations_week,
            },
            'case_stats': {
                'total_analyses': total_analyses,
                'positive_cases': positive_cases,
                'negative_cases': negative_cases,
                'pending_reviews': pending_reviews,
                'analyses_today': analyses_today,
            },
            'stage_distribution': stage_distribution,
            'monthly_detection_rate': round(monthly_detection_rate, 2),
            'pending_ai_analysis': 0,
            'registrations_by_day': registrations_by_day,
        })


# Rename the API detail view to avoid conflict



@method_decorator(csrf_exempt, name='dispatch')
class AnalysisApiDetailView(APIView):
    """GET/PATCH /v1/analyze/<analysis_id> - API endpoint for analysis details"""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def get(self, request, analysis_id):
        try:
            analysis = Analysis.objects.get(pk=analysis_id)
        except Analysis.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(AnalysisSerializer(analysis).data)

    def patch(self, request, analysis_id):
        """Update recommended_checkups for an analysis after the doctor reviews the result."""
        try:
            analysis = Analysis.objects.get(pk=analysis_id)
        except Analysis.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)

        checkups = request.data.get('recommended_checkups', [])
        if not isinstance(checkups, list):
            return Response({"error": "recommended_checkups must be a list"}, status=status.HTTP_400_BAD_REQUEST)

        valid = [c for c in checkups if c in RECOMMENDED_CHECKUP_OPTIONS]
        analysis.recommended_checkups = valid
        analysis.save(update_fields=['recommended_checkups'])

        return Response({"message": "Checkups updated", "recommended_checkups": valid})



# ─── Patient Management Views ───────────────────────────────────────────────────

class PatientManagementView(View):
    """GET /patients/ - Patient management page for all users"""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/patients/')

        user_role = request.user.profile.role
        is_admin = user_role == 'admin' or request.user.is_superuser

        from django.db.models import Count, Max

        if is_admin:
            # Admins see all patients
            patients = Patient.objects.select_related(
                'created_by', 'created_by__profile'
            ).annotate(
                analyses_count=Count('analyses'),
                latest_analysis=Max('analyses__created_at')
            ).order_by('-latest_analysis', '-created_at')
        elif user_role == 'senior_technician':
            # Technicians see patients assigned to them
            patients = Patient.objects.select_related(
                'created_by', 'created_by__profile'
            ).filter(
                assigned_technician=request.user
            ).annotate(
                analyses_count=Count('analyses'),
                latest_analysis=Max('analyses__created_at')
            ).distinct().order_by('-latest_analysis', '-created_at')
        else:
            # Doctors only see patients assigned to them
            patients = Patient.objects.select_related(
                'created_by', 'created_by__profile'
            ).filter(
                assigned_doctor=request.user
            ).annotate(
                analyses_count=Count('analyses'),
                latest_analysis=Max('analyses__created_at')
            ).distinct().order_by('-latest_analysis', '-created_at')

        # Statistics
        total_patients = patients.count()
        from django.utils import timezone
        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_patients = patients.filter(created_at__gte=seven_days_ago).count()

        # Count patients with analyses
        patients_with_analyses = patients.filter(analyses_count__gt=0).count()
        patients_awaiting_analyses = total_patients - patients_with_analyses

        from django.core.paginator import Paginator
        paginator = Paginator(patients, 25)
        page_number = request.GET.get('page')
        patients_page = paginator.get_page(page_number)

        context = {
            'patients': patients_page,
            'total_patients': total_patients,
            'recent_patients': recent_patients,
            'patients_with_analyses': patients_with_analyses,
            'patients_awaiting_analyses': patients_awaiting_analyses,
            'user_role': request.user.profile.role,
        }
        
        # Choose role-specific template
        if user_role == 'user':
            template_name = "doctor/doctor_patients.html"
        elif user_role == 'senior_technician':
            template_name = "technician/technician_patients.html"
        else:
            template_name = "patients.html"

        return render(request, template_name, context)


class AddPatientView(View):
    """GET /patient/add/ - Add new patient page"""
    def get(self, request, patient_id=None):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/patient/add/')

        patient = None
        if patient_id:
            try:
                patient = Patient.objects.get(patient_id=patient_id)
            except Patient.DoesNotExist:
                return redirect('/admin/patients/')

        return render(request, "superuser/add_patient.html", {'patient': patient})

    def post(self, request, patient_id=None):
        """Handle both create and update"""
        if not request.user.is_authenticated:
            return redirect('/login/')

        if patient_id:
            # Update existing patient
            try:
                patient = Patient.objects.get(patient_id=patient_id)
            except Patient.DoesNotExist:
                return redirect('/admin/patients/')

            # Update patient fields
            patient.first_name = request.POST.get('first_name', patient.first_name)
            patient.last_name = request.POST.get('last_name', patient.last_name)
            patient.date_of_birth = request.POST.get('date_of_birth') or patient.date_of_birth
            patient.age = request.POST.get('age') or patient.age
            patient.gender = request.POST.get('gender', patient.gender)
            patient.phone = request.POST.get('phone') or patient.phone
            patient.email = request.POST.get('email') or patient.email
            patient.address = request.POST.get('address') or patient.address
            patient.hpv_status = request.POST.get('hpv_status', patient.hpv_status)
            patient.last_screening_date = request.POST.get('last_screening_date') or patient.last_screening_date
            patient.pregnancy_status = request.POST.get('pregnancy_status', patient.pregnancy_status)
            patient.medical_history = request.POST.get('medical_history') or patient.medical_history
            patient.medications = request.POST.get('medications') or patient.medications
            patient.allergies = request.POST.get('allergies') or patient.allergies
            patient.family_history = request.POST.get('family_history') or patient.family_history
            patient.notes = request.POST.get('notes') or patient.notes
            patient.save()

            # Handle password change / account creation
            new_password = request.POST.get('patient_password')
            if new_password:
                if patient.user_account:
                    # Update password on existing linked account
                    patient.user_account.set_password(new_password)
                    patient.user_account.save()
                    messages.success(request, f'Patient "{patient.full_name}" updated and password changed successfully!')
                elif patient.email or patient.phone:
                    # No linked account yet — create one now
                    username = patient.email if patient.email else patient.phone
                    if not User.objects.filter(username=username).exists():
                        user = User.objects.create_user(
                            username=username,
                            email=patient.email or '',
                            password=new_password,
                            first_name=patient.first_name,
                            last_name=patient.last_name,
                        )
                        user.profile.role = "patient"
                        user.profile.phone = patient.phone
                        user.profile.save()
                        patient.user_account = user
                        patient.save()
                        messages.success(request, f'Patient "{patient.full_name}" updated and login account created (Username: {username})')
                    else:
                        messages.warning(request, f'Patient "{patient.full_name}" updated but username "{username}" already exists.')
                else:
                    messages.warning(request, f'Patient "{patient.full_name}" updated but no email/phone provided to create a login account.')
            else:
                messages.success(request, f'Patient "{patient.full_name}" updated successfully!')

            return redirect(f'/admin/patient/edit/{patient_id}/')



        else:
            # Create new patient
            data = {
                'first_name': request.POST.get('first_name'),
                'last_name': request.POST.get('last_name'),
                'date_of_birth': request.POST.get('date_of_birth') or None,
                'age': request.POST.get('age') or None,
                'gender': request.POST.get('gender'),
                'phone': request.POST.get('phone') or None,
                'email': request.POST.get('email') or None,
                'address': request.POST.get('address') or None,
                'hpv_status': request.POST.get('hpv_status') or 'unknown',
                'last_screening_date': request.POST.get('last_screening_date') or None,
                'pregnancy_status': request.POST.get('pregnancy_status') or 'unknown',
                'medical_history': request.POST.get('medical_history') or None,
                'medications': request.POST.get('medications') or None,
                'allergies': request.POST.get('allergies') or None,
                'family_history': request.POST.get('family_history') or None,
                'notes': request.POST.get('notes') or None,
            }

            patient = Patient.objects.create(**data)
            if request.user.is_authenticated:
                patient.created_by = request.user
                patient.save()

            # Create user account if password is provided
            password = request.POST.get('patient_password')
            if password and (patient.email or patient.phone):
                username = patient.email if patient.email else patient.phone

                if not User.objects.filter(username=username).exists():
                    user = User.objects.create_user(
                        username=username,
                        email=patient.email or '',
                        password=password,
                        first_name=patient.first_name,
                        last_name=patient.last_name,
                    )
                    user.profile.role = "patient"
                    user.profile.phone = patient.phone
                    user.profile.save()

                    patient.user_account = user
                    patient.save()

                    messages.success(request, f'Patient "{patient.full_name}" added successfully! Login account created (Username: {username})')
                else:
                    messages.warning(request, f'Patient "{patient.full_name}" added but user account already exists with username "{username}"')
            else:
                messages.success(request, f'Patient "{patient.full_name}" added successfully!')

            return redirect('/admin/patients/')
        
class PatientProfileView(View):
    """GET/POST /patient/<patient_id> - Patient profile with analysis history and edit functionality"""
    def get(self, request, patient_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/patient/' + str(patient_id))

        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return redirect('/patients/')

        # Access control: each role may only open patients within its scope.
        user_role = request.user.profile.role if hasattr(request.user, 'profile') else None
        if request.user.is_superuser or user_role == 'admin':
            pass  # admins can view any patient
        elif user_role in ('user', 'senior_technician'):
            if not get_patient_queryset_for_user(request.user).filter(patient_id=patient_id).exists():
                messages.error(request, "This patient is not assigned to you.")
                return redirect('/patients/')
        elif user_role == 'patient':
            if patient.user_account_id != request.user.id:
                messages.error(request, "Access denied.")
                return redirect('/patient/dashboard/')

        analyses = Analysis.objects.filter(
            patient=patient
        ).filter(
            Q(is_batch_primary=True) | Q(batch_id__isnull=True)
        ).order_by('-created_at').distinct()

        # Date filter
        date_from = request.GET.get('date_from', '').strip()
        date_to = request.GET.get('date_to', '').strip()
        if date_from:
            try:
                from datetime import datetime
                analyses = analyses.filter(created_at__date__gte=datetime.strptime(date_from, '%Y-%m-%d').date())
            except ValueError:
                pass
        if date_to:
            try:
                from datetime import datetime
                analyses = analyses.filter(created_at__date__lte=datetime.strptime(date_to, '%Y-%m-%d').date())
            except ValueError:
                pass

        total_analyses = analyses.count()
        if total_analyses > 0:
            avg_confidence_val = sum(a.confidence for a in analyses) / total_analyses
            latest_analysis = analyses.first()
        else:
            avg_confidence_val = 0
            latest_analysis = None

        from django.core.paginator import Paginator
        paginator = Paginator(analyses, 10)
        analyses_page = paginator.get_page(request.GET.get('page'))

        # Enrich only current page items
        for analysis in analyses_page:
            if analysis.is_batch_primary and analysis.batch_aggregated_data:
                agg_data = analysis.batch_aggregated_data.get('averaged_prediction', {})
                analysis.confidence_pct = round(agg_data.get('confidence', analysis.confidence) * 100, 1)
                analysis.display_confidence = agg_data.get('confidence', analysis.confidence)
                aggregated_probs = agg_data.get('probabilities', {})
                if aggregated_probs:
                    analysis.probabilities_pct = {k: round(v * 100, 1) for k, v in aggregated_probs.items()}
                elif analysis.probabilities:
                    analysis.probabilities_pct = {k: round(v * 100, 1) for k, v in analysis.probabilities.items()}
                analysis.batch_count = analysis.batch_total_count or 1
            else:
                analysis.confidence_pct = round(analysis.confidence * 100, 1)
                analysis.display_confidence = analysis.confidence
                if analysis.probabilities:
                    analysis.probabilities_pct = {k: round(v * 100, 1) for k, v in analysis.probabilities.items()}
                analysis.batch_count = 1

        # Risk trend chart — all analyses oldest→newest (unfiltered by date range)
        import json as _json
        all_for_chart = list(
            Analysis.objects.filter(
                patient=patient
            ).filter(
                Q(batch_id__isnull=True) | Q(is_batch_primary=True)
            ).order_by('created_at').values('predicted_class', 'predicted_label', 'created_at')
        )
        chart_dates  = _json.dumps([a['created_at'].strftime('%d %b %Y') for a in all_for_chart])
        chart_classes = _json.dumps([a['predicted_class'] for a in all_for_chart])
        chart_labels_map = _json.dumps({
            0: 'Normal', 1: 'ASC-US', 2: 'LSIL', 3: 'HSIL', 4: 'Carcinoma'
        })

        # Clinical notes
        clinical_notes = ClinicalNote.objects.filter(patient=patient).select_related('created_by', 'analysis')

        # Pap smear tests (by technician) and checkup recommendations (by doctor)
        pap_smears = patient.pap_smear_tests.select_related('technician').order_by('-test_date')
        checkups = patient.checkup_recommendations.select_related('doctor', 'pap_smear_test').order_by('-created_at')

        context = {
            'patient': patient,
            'analyses': analyses_page,
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence_val * 100, 2),
            'latest_analysis': latest_analysis,
            'user_role': request.user.profile.role,
            'date_from': date_from,
            'date_to': date_to,
            'chart_dates': chart_dates,
            'chart_classes': chart_classes,
            'chart_labels_map': chart_labels_map,
            'clinical_notes': clinical_notes,
            'pap_smears': pap_smears,
            'checkups': checkups,
        }
        
        # Choose role-specific template
        user_role = request.user.profile.role if hasattr(request.user, 'profile') else None
        if user_role == 'user':
            template_name = "doctor/doctor_patient_profile.html"
        elif user_role == 'senior_technician':
            template_name = "technician/technician_patient_profile.html"
        else:
            template_name = "patient/patient_profile.html"

        return render(request, template_name, context)

    def post(self, request, patient_id):
        """Handle patient update via POST"""
        if not request.user.is_authenticated:
            return redirect('/login/?next=/patient/' + str(patient_id))

        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return redirect('/patients/')

        # Update patient fields from POST data
        patient.first_name = request.POST.get('first_name', patient.first_name)
        patient.last_name = request.POST.get('last_name', patient.last_name)
        patient.date_of_birth = request.POST.get('date_of_birth') or patient.date_of_birth
        patient.gender = request.POST.get('gender', patient.gender)
        patient.phone = request.POST.get('phone', patient.phone) or None
        patient.email = request.POST.get('email', patient.email) or None
        patient.address = request.POST.get('address', patient.address) or None
        patient.hpv_status = request.POST.get('hpv_status', patient.hpv_status)
        patient.pregnancy_status = request.POST.get('pregnancy_status', patient.pregnancy_status)
        patient.medical_history = request.POST.get('medical_history', patient.medical_history) or None
        patient.medications = request.POST.get('medications', patient.medications) or None
        patient.allergies = request.POST.get('allergies', patient.allergies) or None
        patient.family_history = request.POST.get('family_history', patient.family_history) or None
        patient.notes = request.POST.get('notes', patient.notes) or None

        # Handle last_screening_date separately (can be empty string)
        last_screening = request.POST.get('last_screening_date')
        if last_screening:
            patient.last_screening_date = last_screening
        else:
            patient.last_screening_date = None

        patient.save()

        # Add success message
        messages.success(request, f'Patient "{patient.full_name}" updated successfully!')

        # Redirect back to patient profile
        return redirect('/patient/' + str(patient_id))


class AddClinicalNoteView(View):
    """POST /patient/<patient_id>/notes/add/ — add a clinical note for a patient."""
    def post(self, request, patient_id):
        if not request.user.is_authenticated:
            return redirect('/login/')
        role = request.user.profile.role
        if role not in ('admin', 'user', 'senior_technician'):
            return redirect('/dashboard/')

        patient = get_object_or_404(Patient, patient_id=patient_id)
        note_text = request.POST.get('note', '').strip()
        analysis_id = request.POST.get('analysis_id', '').strip()

        if note_text:
            analysis = None
            if analysis_id:
                try:
                    analysis = Analysis.objects.get(analysis_id=analysis_id)
                except Analysis.DoesNotExist:
                    pass
            ClinicalNote.objects.create(
                patient=patient,
                analysis=analysis,
                created_by=request.user,
                note=note_text,
            )
            messages.success(request, 'Clinical note added.')
        else:
            messages.error(request, 'Note cannot be empty.')

        return redirect(f'/patient/{patient_id}#notes')


# ─── Authentication Views (API) ────────────────────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(csrf_exempt, name='dispatch')
class LoginView(APIView):
    """POST /v1/auth/login - Login user and create session"""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def post(self, request):
        from django.contrib.auth import get_user_model
        User = get_user_model()

        login_field = request.data.get("username") or request.data.get("email")
        password = request.data.get("password")

        if not login_field or not password:
            return Response(
                {"error": "Username/Email and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if login_field is an email or username
        # Support email login by looking up the user first
        if "@" in login_field:
            # Email login
            try:
                user_obj = User.objects.get(email=login_field)
                username = user_obj.username
            except User.DoesNotExist:
                return Response(
                    {"error": "Invalid credentials"},
                    status=status.HTTP_401_UNAUTHORIZED
                )
        else:
            # Username login
            username = login_field

        user = authenticate(username=username, password=password)

        if user is None:
            return Response(
                {"error": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        # Create Django session for template authentication
        from django.contrib.auth import login
        login(request, user)

        # Generate tokens
        refresh = RefreshToken.for_user(user)

        # Redirect based on user role
        if user.is_superuser:
            redirect_url = '/admin/'
        elif user.profile.role == 'patient':
            redirect_url = '/patient/dashboard/'
        elif user.profile.role == 'senior_technician':
            redirect_url = '/technician/dashboard/'
        else:
            redirect_url = '/dashboard/'

        return Response({
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.get_full_name(),
                "role": user.profile.role,
            },
            "redirect_url": redirect_url,
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }
        }, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name='dispatch')
class LogoutView(APIView):
    """POST /v1/auth/logout - Logout user and clear session"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # Blacklist refresh token
            refresh_token = request.data.get("refresh")
            if refresh_token:
                token = RefreshToken(refresh_token)
                token.blacklist()

            # Clear Django session
            from django.contrib.auth import logout
            logout(request)

            return Response(
                {"message": "Logout successful"},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            # Still clear session even if token blacklist fails
            from django.contrib.auth import logout
            logout(request)

            return Response(
                {"error": "Invalid token"},
                status=status.HTTP_400_BAD_REQUEST
            )


class TokenRefreshView(APIView):
    """POST /api/v1/token/refresh/ - Refresh access token using refresh token"""
    permission_classes = [AllowAny]

    def post(self, request):
        from rest_framework_simplejwt.tokens import RefreshToken

        refresh = request.data.get("refresh")
        if not refresh:
            return Response(
                {"error": "Refresh token is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            refresh_token = RefreshToken(refresh)
            access_token = refresh_token.access_token

            return Response({
                "access": str(access_token),
                "refresh": str(refresh_token) if getattr(settings, 'SIMPLE_JWT', {}).get('ROTATE_REFRESH_TOKENS', False) else None
            })
        except Exception as e:
            return Response(
                {"error": "Invalid or expired refresh token"},
                status=status.HTTP_401_UNAUTHORIZED
            )


class UserProfileView(APIView):
    """GET /api/v1/auth/profile - Get current user profile"""
    permission_classes = [AllowAny]  # Allow demo users

    def get(self, request):
        # Check if user is authenticated
        if request.user.is_authenticated:
            user = request.user
            return Response({
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.get_full_name(),
                    "role": user.profile.role,
                    "phone": user.profile.phone,
                    "department": user.profile.department,
                    "license_number": user.profile.license_number,
                    "date_joined": user.date_joined,
                    "last_login": user.last_login,
                }
            })
        else:
            # Return demo user data for non-authenticated users
            return Response({
                "user": {
                    "id": "demo",
                    "username": "demo_user",
                    "email": "demo@cervistage.ai",
                    "full_name": "Demo User",
                    "role": "user",
                    "phone": "",
                    "department": "Pathology",
                    "license_number": "DEMO-12345",
                    "date_joined": "2026-04-01T00:00:00Z",
                    "last_login": None,
                }
            })


class ChangePasswordView(APIView):
    """POST /v1/auth/change-password - Change user password"""
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request):
        old_password = request.data.get("old_password")
        new_password = request.data.get("new_password")

        if not old_password or not new_password:
            return Response(
                {"error": "Old password and new password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = request.user
        if not user.check_password(old_password):
            return Response(
                {"error": "Invalid old password"},
                status=status.HTTP_400_BAD_REQUEST
            )

        user.set_password(new_password)
        user.save()

        return Response({
            "message": "Password changed successfully"
        }, status=status.HTTP_200_OK)


# ─── Patient Management Views (API) ─────────────────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
class PatientListCreateAPIView(APIView):
    """GET/POST /api/v1/patients - List or create patients"""
    permission_classes = [AllowAny]  # Explicitly allow all requests
    parser_classes = [JSONParser, MultiPartParser]

    def get(self, request):
        """List all patients"""
        django_user = getattr(request._request, 'user', request.user)
        is_authenticated = django_user.is_authenticated
        user = django_user
        
        if not is_authenticated:
            try:
                from rest_framework_simplejwt.authentication import JWTAuthentication
                jwt_auth = JWTAuthentication()
                auth_header = request.META.get('HTTP_AUTHORIZATION', '')
                if auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    validated_token = jwt_auth.get_validated_token(token)
                    user = jwt_auth.get_user(validated_token)
                    is_authenticated = True
            except Exception:
                pass
                
        if not is_authenticated:
            return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
        patients = get_patient_queryset_for_user(user).order_by('-created_at')

        serializer = PatientListSerializer(patients, many=True)
        return Response({
            "patients": serializer.data,
            "total": patients.count()
        }, status=status.HTTP_200_OK)

    def post(self, request):
        """Create a new patient (admin only)"""
        # Check authentication (supports both session and JWT)
        is_authenticated = request.user.is_authenticated

        # If not authenticated via session, check JWT token
        if not is_authenticated:
            try:
                from rest_framework_simplejwt.authentication import JWTAuthentication
                jwt_auth = JWTAuthentication()
                try:
                    # Manually authenticate with JWT
                    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
                    if auth_header.startswith('Bearer '):
                        token = auth_header.split(' ')[1]
                        from rest_framework_simplejwt.tokens import AccessToken
                        access_token = AccessToken(token)
                        user_id = access_token['user_id']
                        from django.contrib.auth import get_user_model
                        User = get_user_model()
                        request.user = User.objects.get(id=user_id)
                        is_authenticated = True
                except Exception:
                    pass
            except:
                pass

        if not is_authenticated:
            return Response({
                "error": "Authentication required. Please login again."
            }, status=status.HTTP_401_UNAUTHORIZED)

        # Check if user is admin
        try:
            user_role = request.user.profile.role
            is_admin = user_role == 'admin' or request.user.is_superuser
        except Exception:
            is_admin = False

        if not is_admin:
            return Response({
                "error": "Only administrators can add patients"
            }, status=status.HTTP_403_FORBIDDEN)

        serializer = PatientSerializer(data=request.data)
        if serializer.is_valid():
            # Set created_by to the authenticated user if available
            patient = serializer.save()
            if request.user.is_authenticated:
                patient.created_by = request.user
                patient.save()
            return Response({
                "message": "Patient created successfully",
                "patient": PatientSerializer(patient).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            "error": "Invalid data",
            "details": serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


@method_decorator(csrf_exempt, name='dispatch')
class PatientDetailAPIView(APIView):
    """GET/PUT/DELETE /api/v1/patients/<patient_id> - Get, update or delete a patient"""
    permission_classes = [AllowAny]  # Explicitly allow all requests
    parser_classes = [JSONParser]
    authentication_classes = []  # Disable authentication to bypass CSRF enforcement

    def get(self, request, patient_id):
        try:
            # Allow viewing any patient (not just those created by current user)
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({
                "error": "Patient not found"
            }, status=status.HTTP_404_NOT_FOUND)

        serializer = PatientSerializer(patient)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, patient_id):
        try:
            # Allow editing any patient (not just those created by current user)
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({
                "error": "Patient not found"
            }, status=status.HTTP_404_NOT_FOUND)

        serializer = PatientSerializer(patient, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                "message": "Patient updated successfully",
                "patient": PatientSerializer(patient).data
            }, status=status.HTTP_200_OK)
        return Response({
            "error": "Invalid data",
            "details": serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, patient_id):
        try:
            # Allow deleting any patient (not just those created by current user)
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({
                "error": "Patient not found"
            }, status=status.HTTP_404_NOT_FOUND)

        patient_name = patient.full_name
        patient.delete()
        return Response({
            "message": f"Patient '{patient_name}' deleted successfully"
        }, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name='dispatch')
class PatientHistoryAPIView(APIView):
    """GET /api/v1/patients/<patient_id>/history - Get all analyses for a patient"""
    permission_classes = [AllowAny]  # Explicitly allow all requests
    authentication_classes = []  # Disable authentication to bypass CSRF enforcement

    def get(self, request, patient_id):
        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({
                "error": "Patient not found"
            }, status=status.HTTP_404_NOT_FOUND)

        analyses = Analysis.objects.filter(
            patient=patient
        ).order_by('-created_at')

        # Calculate statistics
        total_analyses = analyses.count()
        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses

            # Severity distribution
            severity_counts = {}
            for analysis in analyses:
                label = analysis.predicted_label
                severity_counts[label] = severity_counts.get(label, 0) + 1
        else:
            avg_confidence = 0
            severity_counts = {}

        # Serialize analyses with full details
        analyses_data = []
        for analysis in analyses:
            analyses_data.append({
                "analysis_id": str(analysis.analysis_id),
                "predicted_class": analysis.predicted_class,
                "predicted_label": analysis.predicted_label,
                "confidence": round(analysis.confidence * 100, 2),
                "confidence_level": analysis.confidence_level,
                "recommendation": analysis.recommendation,
                "image_url": analysis.image.url if analysis.image else None,
                "created_at": analysis.created_at.isoformat(),
            })

        return Response({
            "patient": PatientListSerializer(patient).data,
            "statistics": {
                "total_analyses": total_analyses,
                "avg_confidence": round(avg_confidence * 100, 2),
                "severity_distribution": severity_counts
            },
            "analyses": analyses_data
        }, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name='dispatch')
class PatientStatsAPIView(APIView):
    """GET /api/v1/patients/<patient_id>/stats - Get trend statistics for a patient"""
    permission_classes = [AllowAny]  # Explicitly allow all requests
    authentication_classes = []  # Disable authentication to bypass CSRF enforcement

    def get(self, request, patient_id):
        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({
                "error": "Patient not found"
            }, status=status.HTTP_404_NOT_FOUND)

        analyses = Analysis.objects.filter(
            patient=patient
        ).order_by('created_at')

        # Prepare trend data
        trend_data = []
        for analysis in analyses:
            trend_data.append({
                "date": analysis.created_at.isoformat(),
                "predicted_class": analysis.predicted_class,
                "predicted_label": analysis.predicted_label,
                "confidence": round(analysis.confidence * 100, 2),
            })

        # Calculate trend summary
        if len(trend_data) >= 2:
            first_result = trend_data[0]
            last_result = trend_data[-1]
            trend_direction = "improving" if last_result["predicted_class"] < first_result["predicted_class"] else "worsening" if last_result["predicted_class"] > first_result["predicted_class"] else "stable"
        else:
            trend_direction = "insufficient_data"

        return Response({
            "patient_id": str(patient.patient_id),
            "patient_name": patient.full_name,
            "total_analyses": len(trend_data),
            "trend_direction": trend_direction,
            "trend_data": trend_data
        }, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name='dispatch')
class CreatePatientUserView(APIView):
    """POST /api/v1/patient/<patient_id>/create-user - Create user account for existing patient"""
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request, patient_id):
        # Only doctors/admin can create patient accounts
        user_role = request.user.profile.role
        if user_role == 'patient':
            return Response({"error": "Unauthorized"}, status=status.HTTP_403_FORBIDDEN)

        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({"error": "Patient not found"}, status=status.HTTP_404_NOT_FOUND)

        # Check if patient already has a user account
        if patient.user_account:
            return Response({"error": "Patient already has a user account"}, status=status.HTTP_400_BAD_REQUEST)

        password = request.data.get("password")

        if not password:
            return Response({"error": "Password is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Use email as username if available, otherwise use phone
        username = patient.email if patient.email else patient.phone

        if not username:
            return Response({"error": "Patient must have email or phone number for login"}, status=status.HTTP_400_BAD_REQUEST)

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            return Response({"error": f"An account with username '{username}' already exists"}, status=status.HTTP_400_BAD_REQUEST)

        # Create user with 'patient' role
        user = User.objects.create_user(
            username=username,
            email=patient.email or '',
            password=password,
            first_name=patient.first_name,
            last_name=patient.last_name,
        )
        user.profile.role = "patient"
        user.profile.phone = patient.phone
        user.profile.save()

        # Link user to patient
        patient.user_account = user
        patient.save()

        return Response({
            "message": "Patient user account created successfully",
            "username": username,
            "patient_id": str(patient.patient_id)
        }, status=status.HTTP_201_CREATED)


@method_decorator(csrf_exempt, name='dispatch')
class UserManagementView(APIView):
    """GET/PUT/DELETE /api/v1/auth/user/<user_id> - User management endpoint (admin only)"""
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]

    def get(self, request, user_id):
        """Get user details"""
        if request.user.profile.role != 'admin':
            return Response(
                {"error": "Admin access required"},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            target_user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response(
                {"error": "User not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        return Response({
            "user": {
                "id": target_user.id,
                "username": target_user.username,
                "email": target_user.email,
                "full_name": target_user.get_full_name(),
                "role": target_user.profile.role,
                "is_active": target_user.is_active,
                "date_joined": target_user.date_joined,
                "phone": target_user.profile.phone,
                "department": target_user.profile.department,
                "license_number": target_user.profile.license_number,
            }
        })

    def put(self, request, user_id):
        """Update user details"""
        if request.user.profile.role != 'admin':
            return Response(
                {"error": "Admin access required"},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            target_user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response(
                {"error": "User not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Prevent admins from editing other admins
        if target_user.profile.role == 'admin' and target_user != request.user:
            return Response(
                {"error": "Cannot edit other admin users"},
                status=status.HTTP_403_FORBIDDEN
            )

        # Get fields to update
        username = request.data.get("username")
        email = request.data.get("email")
        full_name = request.data.get("full_name")
        role = request.data.get("role")
        is_active = request.data.get("is_active")

        if username:
            if User.objects.filter(username=username).exclude(id=user_id).exists():
                return Response(
                    {"error": "Username already exists"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            target_user.username = username

        if email:
            if User.objects.filter(email=email).exclude(id=user_id).exists():
                return Response(
                    {"error": "Email already exists"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            target_user.email = email

        if full_name:
            name_parts = full_name.split()
            target_user.first_name = name_parts[0] if name_parts else ""
            target_user.last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        if role and target_user.profile.role != 'admin':
            valid_roles = ['user']
            if role in valid_roles:
                target_user.profile.role = role
                target_user.profile.save()

        if is_active is not None:
            target_user.is_active = bool(is_active)

        target_user.save()

        return Response({
            "message": "User updated successfully",
            "user": {
                "id": target_user.id,
                "username": target_user.username,
                "email": target_user.email,
                "full_name": target_user.get_full_name(),
                "role": target_user.profile.role,
                "is_active": target_user.is_active,
            }
        })

    def delete(self, request, user_id):
        """Delete user"""
        if request.user.profile.role != 'admin':
            return Response(
                {"error": "Admin access required"},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            target_user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response(
                {"error": "User not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        if target_user.profile.role == 'admin':
            return Response(
                {"error": "Cannot delete admin users"},
                status=status.HTTP_403_FORBIDDEN
            )

        if target_user == request.user:
            return Response(
                {"error": "Cannot delete your own account"},
                status=status.HTTP_400_BAD_REQUEST
            )

        username = target_user.username
        target_user.delete()

        return Response({
            "message": f"User '{username}' deleted successfully"
        }, status=status.HTTP_200_OK)


# ─── Admin Views ───────────────────────────────────────────────────────────────

def generate_results_frames(analysis_records, predictions_list):
    """
    Generates individual overlaid frames from the analyzed images.
    Returns a list of URLs of the generated frames.
    """
    if not analysis_records:
        return []
        
    anim_dir_name = f"anim_{uuid.uuid4().hex[:8]}"
    anim_dir_path = os.path.join(settings.MEDIA_ROOT, 'animations', anim_dir_name)
    os.makedirs(anim_dir_path, exist_ok=True)
    safe_print(f"[DEBUG] Generating frames in: {anim_dir_path}")
    
    width, height = 640, 640
    frame_urls = []
    
    for i, (record, pred) in enumerate(zip(analysis_records, predictions_list)):
        try:
            img_path = record.image.path
            safe_print(f"[DEBUG] Frame {i}: Reading image from {img_path}")
            if not os.path.exists(img_path):
                safe_print(f"[WARN] Image file not found: {img_path}")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                safe_print(f"[WARN] Failed to read image with cv2: {img_path}")
                continue
            
            h, w = img.shape[:2]
            scale = min(width/w, height/h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            y_offset = (height - new_h) // 2
            x_offset = (width - new_w) // 2
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            label = pred["predicted_label"]
            conf = f"Confidence: {round(pred['confidence'] * 100)}%"
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            colors = {
                0: (94, 197, 34),    # Green (BGR)
                1: (22, 204, 132),   # Lime
                2: (8, 179, 234),    # Yellow
                3: (22, 115, 249),   # Orange
                4: (68, 68, 239)     # Red
            }
            cls_idx = pred["predicted_class"]
            color = colors.get(cls_idx, (255, 255, 255))
            
            cv2.putText(frame, label, (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, conf, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            frame_filename = f"frame_{i:03d}.jpg"
            frame_filepath = os.path.join(anim_dir_path, frame_filename)
            cv2.imwrite(
                frame_filepath,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            safe_print(f"[DEBUG] Frame {i}: Saved to {frame_filepath}")
            
            frame_url = f"{settings.MEDIA_URL}animations/{anim_dir_name}/{frame_filename}"
            frame_urls.append(frame_url)
            safe_print(f"[DEBUG] Frame {i}: URL = {frame_url}")
            
        except Exception as e:
            safe_print(f"[ERROR] Frame {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    safe_print(f"[DEBUG] Generated {len(frame_urls)} frames total")
    return frame_urls


# ── Shared helpers for the analyze endpoints ────────────────────────────────
ANALYZE_MAX_FILES = 200
ANALYZE_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}


def analyze_guard(request, files, max_files=ANALYZE_MAX_FILES):
    """Shared pre-checks for the analyze endpoints. Returns an error Response or None."""
    if request.user.profile.role == 'patient':
        return Response(
            {"error": "Patients cannot perform analysis. Please contact your healthcare provider."},
            status=status.HTTP_403_FORBIDDEN,
        )
    if len(files) > max_files:
        return Response({"error": f"Maximum {max_files} images allowed"}, status=400)
    if not files:
        return Response({"error": "No image files provided"}, status=status.HTTP_400_BAD_REQUEST)
    return None


def store_batch_aggregated(analysis_records, averaged_prediction, predictions_list, files, recommended_checkups):
    """Persist averaged batch results onto the primary analysis record (multi-image only)."""
    if len(files) > 1 and analysis_records:
        primary = analysis_records[0]
        primary.batch_aggregated_data = {
            "averaged_prediction": {
                "predicted_class": averaged_prediction["predicted_class"],
                "predicted_label": averaged_prediction["predicted_label"],
                "stage_label": averaged_prediction.get("stage_label"),
                "probabilities": averaged_prediction["probabilities"],
                "confidence": averaged_prediction["confidence"],
                "confidence_level": averaged_prediction.get("confidence_level"),
                "risk_level": averaged_prediction.get("risk_level"),
                "risk_color": averaged_prediction.get("risk_color"),
                "recommendation": averaged_prediction.get("recommendation"),
                "explanation": averaged_prediction.get("explanation"),
            },
            "images_analyzed": len(predictions_list),
            "images_total": len(files),
            "individual_analysis_ids": [str(a.analysis_id) for a in analysis_records],
            "recommended_checkups": recommended_checkups,
        }
        primary.save(update_fields=['batch_aggregated_data'])


def build_analyze_response(averaged_prediction, predictions_list, analysis_records, files,
                           recommended_checkups, patient, invalid_images, frame_urls=None):
    """Assemble the shared analyze API response payload."""
    individual_results = [
        {
            "analysis_id": str(record.analysis_id),
            "predicted_class": pred["predicted_class"],
            "predicted_label": pred["predicted_label"],
            "confidence": pred["confidence"],
            "image_url": record.image.url if record.image else None,
        }
        for record, pred in zip(analysis_records, predictions_list)
    ]
    resp = {
        "predicted_class": averaged_prediction["predicted_class"],
        "predicted_label": averaged_prediction["predicted_label"],
        "stage_label": averaged_prediction["stage_label"],
        "probabilities": averaged_prediction["probabilities"],
        "confidence": averaged_prediction["confidence"],
        "confidence_level": averaged_prediction["confidence_level"],
        "risk_level": averaged_prediction["risk_level"],
        "risk_color": averaged_prediction["risk_color"],
        "recommendation": averaged_prediction["recommendation"],
        "explanation": averaged_prediction["explanation"],
        "images_analyzed": len(predictions_list),
        "images_total": len(files),
        "analysis_id": str(analysis_records[0].analysis_id) if analysis_records else None,
        "individual_analyses": [str(a.analysis_id) for a in analysis_records],
        "individual_results": individual_results,
        "recommended_checkups": recommended_checkups,
        "patient": {
            "patient_id": str(patient.patient_id) if patient else None,
            "full_name": patient.full_name if patient else None,
            "age": patient.age if patient else None,
        } if patient else None,
    }
    if frame_urls is not None:
        resp["frame_urls"] = frame_urls
    if invalid_images:
        resp["validation_warnings"] = [
            {"filename": img["filename"], "reason": img["reason"], "score": img["score"]}
            for img in invalid_images
        ]
        resp["validation_warning"] = (
            f"{len(invalid_images)} of {len(files)} images were rejected due to quality validation. "
            "Results are based on the remaining valid images."
        )
    return resp


class AnalyzeView(APIView):
    """POST /v1/analyze — upload multiple images, run AI staging on each, return averaged result."""
    parser_classes = [MultiPartParser]
    permission_classes = [CanAnalyze]

    def post(self, request):
        files = request.FILES.getlist('file')
        guard = analyze_guard(request, files)
        if guard:
            return guard

        safe_print(f"[DEBUG] Received {len(files)} files for analysis")

        allowed_types = ANALYZE_ALLOWED_TYPES
        svc = VPSInferenceService.get()

        if svc.session is None:
            return Response(
                {"error": "Model not loaded. Copy cervistage_net.onnx to the models/ directory."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        recommended_checkups = get_selected_checkups(request)
        batch_id = uuid.uuid4() if len(files) > 1 else None
        patient_id = request.data.get('patient_id')
        patient, patient_error = get_allowed_patient_for_user(request.user, patient_id)
        if patient_error:
            return patient_error

        # ── Phase 1: validate MIME + image quality ───────────────────────
        invalid_images = []
        valid_files = []  # (original_index, image_file)
        for idx, image_file in enumerate(files):
            if image_file.content_type not in allowed_types:
                return Response(
                    {"error": f"File {idx+1} ({image_file.name}): Unsupported type {image_file.content_type}. Allowed: JPEG, PNG, BMP, TIFF"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            validation_result = ImageValidator.validate_file(image_file)
            safe_print(f"[DEBUG] Image {idx+1} validation score: {validation_result['score']}/100")
            if not validation_result["is_valid"]:
                invalid_images.append({
                    "index": idx + 1,
                    "filename": image_file.name,
                    "reason": validation_result["errors"][0][1] if validation_result["errors"] else "Unknown",
                    "score": validation_result["score"],
                })
                safe_print(f"[WARN] Image {idx+1} failed validation: {validation_result['errors']}")
                continue
            image_file.seek(0)  # reset after validation read
            valid_files.append((idx, image_file))

        if not valid_files:
            return Response({
                "error": "No valid images could be analyzed",
                "invalid_images": invalid_images,
                "message": f"All {len(files)} uploaded images failed quality validation. "
                           "Please ensure you are uploading valid cervical cytology/cell sample images.",
            }, status=status.HTTP_400_BAD_REQUEST)

        # ── Phase 2: single batch inference call ─────────────────────────
        try:
            raw_preds = svc.predict_batch([f for _, f in valid_files])
        except RuntimeError as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # ── Phase 3: look up patient once, store DB records ───────────────
        predictions_list = []
        analysis_records = []
        _DB_SKIP = {"stage_label", "risk_level", "risk_color", "explanation",
                    "confidence_interpretation", "heatmap", "inference_time"}

        for (orig_idx, image_file), prediction in zip(valid_files, raw_preds):
            if prediction is None:
                continue
            predictions_list.append(prediction)
            pred_for_db = {k: v for k, v in prediction.items() if k not in _DB_SKIP}
            image_file.seek(0)
            safe_print(f"[DEBUG] Processing file {orig_idx+1}/{len(files)}: {image_file.name}")
            analysis = Analysis.objects.create(
                created_by=request.user,
                patient=patient,
                image=image_file,
                **pred_for_db,
                recommended_checkups=recommended_checkups,
                batch_id=batch_id,
                is_batch_primary=(orig_idx == 0 and len(files) > 1),
                batch_total_count=len(files) if len(files) > 1 else None,
                batch_position=orig_idx + 1 if len(files) > 1 else None,
            )
            analysis_records.append(analysis)
            safe_print(f"[DEBUG] ✓ Stored analysis for file {orig_idx+1}: {analysis.analysis_id}")

        if not predictions_list:
            return Response({"error": "All images failed inference"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Average predictions across all images
        safe_print(f"[DEBUG] Averaging {len(predictions_list)} predictions")
        averaged_prediction = svc.average_predictions(predictions_list)

        # Store aggregated data in the primary analysis if it's a batch
        store_batch_aggregated(analysis_records, averaged_prediction, predictions_list, files, recommended_checkups)

        # Notify the assigned doctor when a technician runs an analysis
        role = getattr(getattr(request.user, 'profile', None), 'role', None)
        if role == 'senior_technician' and patient and patient.assigned_doctor and analysis_records:
            Notification.notify(
                patient.assigned_doctor,
                f"{request.user.get_full_name() or request.user.username} ran an AI analysis for {patient.full_name}",
                url=f"/doctor/analysis/{analysis_records[0].analysis_id}",
            )

        # Generate frames of the results
        frame_urls = []
        if len(files) <= 30:
            frame_urls = generate_results_frames(analysis_records, predictions_list)

        resp = build_analyze_response(
            averaged_prediction, predictions_list, analysis_records, files,
            recommended_checkups, patient, invalid_images, frame_urls=frame_urls,
        )
        return Response(resp, status=status.HTTP_201_CREATED)


@method_decorator(csrf_exempt, name='dispatch')
class FastAnalyzeView(AnalyzeView):
    """POST /api/v1/analyze/fast — same batched inference pipeline as AnalyzeView.
    Kept as a separate endpoint so the frontend's mode selection keeps working."""
    pass


@method_decorator(csrf_exempt, name='dispatch')
class VPSAnalyzeView(AnalyzeView):
    """POST /api/v1/analyze/vps — same batched inference pipeline as AnalyzeView.
    Kept as a separate endpoint so the frontend's mode selection keeps working."""
    pass


class HealthView(APIView):
    """GET /v1/health"""

    def get(self, request):
        model_loaded = VPSInferenceService.get().session is not None
        return Response({"status": "ok", "model_loaded": model_loaded})


class ReportView(APIView):
    """POST /api/v1/reports/generate - generate PDF report for an analysis."""
    parser_classes = [JSONParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        analysis_id = request.data.get("analysis_id")

        try:
            analysis = Analysis.objects.get(pk=analysis_id)
        except Analysis.DoesNotExist:
            return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get user name for the report
        user_name = ""
        if analysis.created_by:
            user_name = analysis.created_by.get_full_name() or analysis.created_by.username

        try:
            pdf_buffer = self._generate_pdf(analysis, user_name)
        except ImportError:
            return Response(
                {"error": "reportlab not installed. Run: pip install reportlab"},
                status=status.HTTP_501_NOT_IMPLEMENTED,
            )

        response = HttpResponse(pdf_buffer.getvalue(), content_type="application/pdf")
        response["Content-Disposition"] = (
            f'attachment; filename="cervistage_report_{str(analysis_id)[:8]}.pdf"'
        )
        return response

    def _generate_pdf(self, analysis, user_name):
        """Generate a clean, professional PDF report for single or batch analysis."""
        import io
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                         Paragraph, Spacer, PageBreak, Image,
                                         HRFlowable, KeepTogether)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.pdfgen import canvas as pdfcanvas

        W, H = A4
        MARGIN = 1.8 * cm
        INNER_W = W - 2 * MARGIN

        buf = io.BytesIO()

        # ── stage palette ──────────────────────────────────────────────────────
        STAGE_BG   = {0:'#dcfce7', 1:'#ecfccb', 2:'#fef9c3', 3:'#ffedd5', 4:'#fee2e2'}
        STAGE_FG   = {0:'#166534', 1:'#365314', 2:'#713f12', 3:'#7c2d12', 4:'#7f1d1d'}
        STAGE_ACCENT = {0:'#22c55e', 1:'#84cc16', 2:'#eab308', 3:'#f97316', 4:'#ef4444'}

        # ── styles ─────────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()

        def S(name, **kw):
            base = kw.pop('parent', styles['Normal'])
            return ParagraphStyle(name, parent=base, **kw)

        hdr_style   = S('Hdr',   fontSize=20, textColor=colors.white,
                        fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=0)
        sub_style   = S('Sub',   fontSize=9,  textColor=colors.HexColor('#cbd5e1'),
                        alignment=TA_CENTER, spaceAfter=0)
        sec_style   = S('Sec',   fontSize=11, textColor=colors.HexColor('#1e3a8a'),
                        fontName='Helvetica-Bold', spaceBefore=0.4*cm, spaceAfter=0.2*cm)
        body_style  = S('Body',  fontSize=9,  textColor=colors.HexColor('#1e293b'))
        small_style = S('Sm',    fontSize=8,  textColor=colors.HexColor('#475569'))
        label_style = S('Lbl',   fontSize=8,  textColor=colors.HexColor('#64748b'),
                        fontName='Helvetica-Bold')
        val_style   = S('Val',   fontSize=9,  textColor=colors.HexColor('#0f172a'))
        center_style= S('Ctr',   fontSize=9,  alignment=TA_CENTER)
        foot_style  = S('Foot',  fontSize=7,  textColor=colors.HexColor('#94a3b8'),
                        alignment=TA_CENTER)

        # ── page header/footer callback ────────────────────────────────────────
        gen_time = analysis.created_at.strftime('%Y-%m-%d %H:%M UTC')

        def on_page(canv, doc):
            canv.saveState()
            # top banner
            canv.setFillColor(colors.HexColor('#1e3a8a'))
            canv.rect(0, H - 2*cm, W, 2*cm, fill=1, stroke=0)
            canv.setFont('Helvetica-Bold', 13)
            canv.setFillColor(colors.white)
            canv.drawString(MARGIN, H - 1.3*cm, 'CerviStage AI — Clinical Report')
            canv.setFont('Helvetica', 8)
            canv.setFillColor(colors.HexColor('#93c5fd'))
            canv.drawRightString(W - MARGIN, H - 1.3*cm,
                                 f'Generated: {gen_time}')
            # bottom line
            canv.setStrokeColor(colors.HexColor('#e2e8f0'))
            canv.setLineWidth(0.5)
            canv.line(MARGIN, 1.2*cm, W - MARGIN, 1.2*cm)
            canv.setFont('Helvetica', 7)
            canv.setFillColor(colors.HexColor('#94a3b8'))
            canv.drawString(MARGIN, 0.6*cm,
                            'For clinical screening assistance only — not a substitute for professional diagnosis.')
            canv.drawRightString(W - MARGIN, 0.6*cm,
                                 f'Page {doc.page}')
            canv.restoreState()

        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=MARGIN, leftMargin=MARGIN,
            topMargin=2.4*cm, bottomMargin=1.8*cm,
            onFirstPage=on_page, onLaterPages=on_page,
        )

        content = []

        # ── helpers ────────────────────────────────────────────────────────────
        def divider():
            return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#e2e8f0'),
                              spaceAfter=0.3*cm, spaceBefore=0.1*cm)

        def section(title):
            return Paragraph(title, sec_style)

        def kv_table(rows, col_w=None):
            """Two-column key-value table."""
            col_w = col_w or [4.5*cm, INNER_W - 4.5*cm]
            data = [[Paragraph(k, label_style), Paragraph(str(v), val_style)] for k, v in rows]
            t = Table(data, colWidths=col_w)
            t.setStyle(TableStyle([
                ('VALIGN',        (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('TOPPADDING',    (0,0), (-1,-1), 3),
                ('ROWBACKGROUNDS',(0,0),(-1,-1),
                 [colors.HexColor('#f8fafc'), colors.white]),
            ]))
            return t

        def colored_badge(text, bg, fg):
            badge = Table([[Paragraph(text, ParagraphStyle(
                'badge', parent=styles['Normal'],
                fontSize=14, fontName='Helvetica-Bold',
                textColor=colors.HexColor(fg), alignment=TA_CENTER))]],
                colWidths=[INNER_W])
            badge.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor(bg)),
                ('ROUNDEDCORNERS',(0,0),(-1,-1), [6,6,6,6]),
                ('TOPPADDING',    (0,0),(-1,-1), 10),
                ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ]))
            return badge

        def prob_bars(probs_dict, predicted_class):
            """Render probability bars as a table."""
            CLASS_COLORS = ['#22c55e','#84cc16','#eab308','#f97316','#ef4444']
            BAR_W = INNER_W - 5*cm - 2*cm
            rows = []
            for cls_idx, (label, prob) in enumerate(probs_dict.items()):
                pct = float(prob) * 100
                fill_w  = max(pct / 100 * BAR_W, 0.01*cm)
                empty_w = max(BAR_W - fill_w, 0.01*cm)
                bar_color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#94a3b8'
                is_pred = (cls_idx == predicted_class)
                bar_row = Table(
                    [[''      , '']],
                    colWidths=[fill_w, empty_w]
                )
                bar_row.setStyle(TableStyle([
                    ('BACKGROUND', (0,0),(0,0), colors.HexColor(bar_color)),
                    ('BACKGROUND', (1,0),(1,0), colors.HexColor('#e2e8f0')),
                    ('TOPPADDING',(0,0),(-1,-1),0),
                    ('BOTTOMPADDING',(0,0),(-1,-1),0),
                    ('LEFTPADDING',(0,0),(-1,-1),0),
                    ('RIGHTPADDING',(0,0),(-1,-1),0),
                ]))
                lbl_para = Paragraph(
                    f'<b>{label}</b>' if is_pred else label,
                    ParagraphStyle('pl', parent=styles['Normal'], fontSize=8,
                                   textColor=colors.HexColor('#0f172a' if is_pred else '#475569'))
                )
                pct_para = Paragraph(
                    f'<b>{pct:.1f}%</b>' if is_pred else f'{pct:.1f}%',
                    ParagraphStyle('pp', parent=styles['Normal'], fontSize=8,
                                   alignment=TA_RIGHT,
                                   textColor=colors.HexColor('#0f172a' if is_pred else '#64748b'))
                )
                rows.append([lbl_para, bar_row, pct_para])

            t = Table(rows, colWidths=[5*cm, BAR_W, 2*cm])
            t.setStyle(TableStyle([
                ('VALIGN',        (0,0),(-1,-1),'MIDDLE'),
                ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                ('TOPPADDING',    (0,0),(-1,-1), 5),
                ('LEFTPADDING',   (1,0),(1,-1), 0),
                ('RIGHTPADDING',  (1,0),(1,-1), 0),
            ]))
            return t

        def safe_image(path, w, h):
            if path and os.path.exists(path):
                try:
                    return Image(path, width=w, height=h, lazy=2)
                except Exception:
                    pass
            return Paragraph('<i>Image unavailable</i>', small_style)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Patient Info + Report Meta
        # ══════════════════════════════════════════════════════════════════════
        content.append(Spacer(1, 0.3*cm))

        patient = analysis.patient

        # Report meta row
        meta_data = [
            [Paragraph('Analysis ID', label_style),
             Paragraph('Date & Time', label_style),
             Paragraph('Analyzed By', label_style)],
            [Paragraph(str(analysis.analysis_id)[:16] + '…', val_style),
             Paragraph(analysis.created_at.strftime('%d %b %Y, %H:%M'), val_style),
             Paragraph(user_name or '—', val_style)],
        ]
        meta_t = Table(meta_data, colWidths=[INNER_W/3]*3)
        meta_t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#f1f5f9')),
            ('BACKGROUND',    (0,1),(-1,1), colors.HexColor('#f8fafc')),
            ('BOX',           (0,0),(-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID',     (0,0),(-1,-1), 0.3, colors.HexColor('#e2e8f0')),
            ('TOPPADDING',    (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ]))
        content.append(meta_t)
        content.append(Spacer(1, 0.5*cm))

        if patient:
            content.append(section('Patient Information'))
            content.append(divider())

            basic = kv_table([
                ('Full Name',    patient.full_name or '—'),
                ('Patient ID',   str(patient.patient_id)[:8] if patient.patient_id else '—'),
                ('Date of Birth',patient.date_of_birth.strftime('%d %b %Y') if patient.date_of_birth else '—'),
                ('Age',          str(patient.age) if patient.age else '—'),
                ('Gender',       dict(patient._meta.get_field('gender').choices).get(patient.gender,'—')),
                ('Phone',        patient.phone or '—'),
                ('Email',        patient.email or '—'),
                ('Address',      patient.address or '—'),
            ])
            content.append(basic)
            content.append(Spacer(1, 0.3*cm))

            content.append(section('Medical Information'))
            content.append(divider())
            content.append(kv_table([
                ('HPV Status',       patient.get_hpv_status_display() or '—'),
                ('Pregnancy Status', patient.get_pregnancy_status_display() or '—'),
                ('Last Screening',   patient.last_screening_date.strftime('%d %b %Y') if patient.last_screening_date else '—'),
            ]))

            extras = [
                ('Medical History',    patient.medical_history),
                ('Current Medications',patient.medications),
                ('Allergies',          patient.allergies),
                ('Family History',     patient.family_history),
                ('Additional Notes',   patient.notes),
            ]
            extras = [(k,v) for k,v in extras if v]
            if extras:
                content.append(Spacer(1, 0.3*cm))
                content.append(section('Clinical History'))
                content.append(divider())
                content.append(kv_table(extras))

        content.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════
        # Determine single vs batch
        # ══════════════════════════════════════════════════════════════════════
        batch_analyses = []
        if analysis.batch_id:
            batch_analyses = list(
                Analysis.objects.filter(batch_id=analysis.batch_id).order_by('batch_position')
            )
            for ba in batch_analyses:
                ba.confidence_pct = round(ba.confidence * 100, 1)
                if ba.probabilities:
                    ba.probabilities_pct = {k: round(float(v)*100, 1) for k,v in ba.probabilities.items()}
                else:
                    ba.probabilities_pct = {}

        is_batch = bool(batch_analyses)

        # ══════════════════════════════════════════════════════════════════════
        # SINGLE IMAGE ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        if not is_batch:
            cls = analysis.predicted_class
            bg  = STAGE_BG.get(cls, '#f1f5f9')
            fg  = STAGE_FG.get(cls, '#0f172a')
            acc = STAGE_ACCENT.get(cls, '#64748b')

            content.append(section('Analysis Result'))
            content.append(divider())

            # Result banner
            content.append(colored_badge(analysis.predicted_label, bg, fg))
            content.append(Spacer(1, 0.3*cm))

            # Image + key metrics side by side
            img_w, img_h = 8*cm, 6*cm
            img_cell = safe_image(
                analysis.image.path if analysis.image else None, img_w, img_h)

            metrics = [
                ('Confidence',  f"{round(analysis.confidence*100)}%"),
                ('AI Level',    analysis.confidence_level),
            ]
            metrics_t = kv_table(metrics, col_w=[4*cm, INNER_W - img_w - 1*cm - 4*cm])

            side_t = Table([[img_cell, metrics_t]],
                           colWidths=[img_w + 0.5*cm, INNER_W - img_w - 0.5*cm])
            side_t.setStyle(TableStyle([
                ('VALIGN', (0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',  (1,0),(1,-1), 0.5*cm),
                ('TOPPADDING',   (0,0),(-1,-1), 0),
                ('BOTTOMPADDING',(0,0),(-1,-1), 0),
            ]))
            content.append(side_t)
            content.append(Spacer(1, 0.5*cm))

            # Probabilities
            if analysis.probabilities:
                content.append(section('Class Probabilities'))
                content.append(divider())
                content.append(prob_bars(analysis.probabilities, cls))
                content.append(Spacer(1, 0.4*cm))

        # ══════════════════════════════════════════════════════════════════════
        # BATCH IMAGE ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        else:
            agg = analysis.batch_aggregated_data or {}
            avg_pred = agg.get('averaged_prediction', {})
            avg_cls  = avg_pred.get('predicted_class', analysis.predicted_class)
            avg_lbl  = avg_pred.get('predicted_label', analysis.predicted_label)
            avg_conf = round(avg_pred.get('confidence', analysis.confidence) * 100)

            bg  = STAGE_BG.get(avg_cls, '#f1f5f9')
            fg  = STAGE_FG.get(avg_cls, '#0f172a')

            content.append(section(f'Batch Analysis Summary — {len(batch_analyses)} Images'))
            content.append(divider())
            content.append(colored_badge(avg_lbl, bg, fg))
            content.append(Spacer(1, 0.2*cm))

            summary_data = [
                ('Averaged Result',  avg_lbl),
                ('Avg Confidence',   f'{avg_conf}%'),
                ('Images Analyzed',  str(len(batch_analyses))),
                ('Batch ID',         str(analysis.batch_id)[:16] + '…'),
            ]
            content.append(kv_table(summary_data))
            content.append(Spacer(1, 0.5*cm))

            # Averaged probabilities
            avg_probs = avg_pred.get('probabilities', {})
            if avg_probs:
                content.append(section('Averaged Class Probabilities'))
                content.append(divider())
                content.append(prob_bars(avg_probs, avg_cls))
                content.append(Spacer(1, 0.4*cm))

            content.append(PageBreak())

            # Individual images — 2 per row
            content.append(section('Individual Image Results'))
            content.append(divider())

            THUMB_W = (INNER_W - 0.5*cm) / 2
            THUMB_H = THUMB_W * 0.7

            for i in range(0, len(batch_analyses), 2):
                pair = batch_analyses[i:i+2]
                cells = []
                for ba in pair:
                    cls_b = ba.predicted_class
                    bg_b  = STAGE_BG.get(cls_b, '#f1f5f9')
                    fg_b  = STAGE_FG.get(cls_b, '#0f172a')
                    acc_b = STAGE_ACCENT.get(cls_b, '#94a3b8')

                    img_el = safe_image(
                        ba.image.path if ba.image else None, THUMB_W, THUMB_H)

                    lbl_t = Table(
                        [[Paragraph(ba.predicted_label,
                           ParagraphStyle('bl', parent=styles['Normal'],
                               fontSize=9, fontName='Helvetica-Bold',
                               textColor=colors.HexColor(fg_b), alignment=TA_CENTER))]],
                        colWidths=[THUMB_W])
                    lbl_t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0),(-1,-1), colors.HexColor(bg_b)),
                        ('TOPPADDING', (0,0),(-1,-1), 4),
                        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
                    ]))

                    conf_t = Paragraph(
                        f'Confidence: <b>{ba.confidence_pct}%</b>',
                        ParagraphStyle('bc', parent=styles['Normal'],
                                       fontSize=8, alignment=TA_CENTER,
                                       textColor=colors.HexColor('#475569')))

                    cell_inner = Table(
                        [[img_el], [lbl_t], [conf_t]],
                        colWidths=[THUMB_W])
                    cell_inner.setStyle(TableStyle([
                        ('VALIGN',        (0,0),(-1,-1),'MIDDLE'),
                        ('TOPPADDING',    (0,0),(-1,-1), 2),
                        ('BOTTOMPADDING', (0,0),(-1,-1), 2),
                        ('BOX', (0,0),(-1,-1), 0.5, colors.HexColor('#e2e8f0')),
                    ]))
                    cells.append(cell_inner)

                # Pad to 2 cols
                while len(cells) < 2:
                    cells.append('')

                row_t = Table([cells], colWidths=[THUMB_W, THUMB_W],
                              hAlign='LEFT', spaceBefore=0.3*cm)
                row_t.setStyle(TableStyle([
                    ('VALIGN',       (0,0),(-1,-1),'TOP'),
                    ('LEFTPADDING',  (0,0),(-1,-1), 0),
                    ('RIGHTPADDING', (0,0),(-1,-1), 0),
                    ('COLPADDING',   (0,0),(0,-1),  0.25*cm),
                ]))
                content.append(row_t)
                content.append(Spacer(1, 0.3*cm))

        # ══════════════════════════════════════════════════════════════════════
        # Clinical Recommendation + Checkups (both single & batch)
        # ══════════════════════════════════════════════════════════════════════
        if analysis.recommendation:
            content.append(section('Clinical Recommendation'))
            content.append(divider())
            rec_t = Table(
                [[Paragraph(analysis.recommendation, body_style)]],
                colWidths=[INNER_W])
            rec_t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#fef9c3')),
                ('BOX',           (0,0),(-1,-1), 1,   colors.HexColor('#fde047')),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ('RIGHTPADDING',  (0,0),(-1,-1), 10),
                ('TOPPADDING',    (0,0),(-1,-1), 8),
                ('BOTTOMPADDING', (0,0),(-1,-1), 8),
            ]))
            content.append(rec_t)
            content.append(Spacer(1, 0.4*cm))

        # Technician's Pap smear report and the doctor's review for this patient
        pap_smear = None
        checkup = None
        if patient:
            from .models import CheckupRecommendation, PapSmearTest
            pap_smear = PapSmearTest.objects.filter(
                patient=patient
            ).select_related('technician').order_by('-test_date').first()
            checkup = CheckupRecommendation.objects.filter(
                patient=patient
            ).select_related('doctor').order_by('-created_at').first()

        if pap_smear:
            content.append(section('Lab Technician Report'))
            content.append(divider())
            content.append(kv_table([
                ('Pap Smear Result', 'Positive (+ve)' if pap_smear.result == '+ve' else 'Negative (-ve)'),
                ('Performed By', pap_smear.technician.get_full_name() if pap_smear.technician else '—'),
                ('Test Date', pap_smear.test_date.strftime('%d %b %Y')),
                ('Technician Notes', pap_smear.notes or '—'),
            ]))
            content.append(Spacer(1, 0.4*cm))

        # Doctor's recommended checkups (fall back to AI-suggested checkups)
        doctor_checkups = checkup.recommended_list() if checkup else list(analysis.recommended_checkups or [])
        if doctor_checkups or (checkup and checkup.treatment_follow_up):
            content.append(section('Doctor Review & Recommended Checkups'))
            content.append(divider())
            if doctor_checkups:
                checkup_cells = [
                    [Paragraph(f'✓  {c}', ParagraphStyle(
                        'ck', parent=styles['Normal'], fontSize=9,
                        textColor=colors.HexColor('#1e40af')))]
                    for c in doctor_checkups
                ]
                ck_t = Table(checkup_cells, colWidths=[INNER_W])
                ck_t.setStyle(TableStyle([
                    ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#eff6ff')),
                    ('BOX',           (0,0),(-1,-1), 0.5, colors.HexColor('#bfdbfe')),
                    ('TOPPADDING',    (0,0),(-1,-1), 5),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                    ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ]))
                content.append(ck_t)
            if checkup and checkup.treatment_follow_up:
                content.append(Spacer(1, 0.2*cm))
                content.append(Paragraph('<b>Treatment &amp; Follow-up Plan</b>', body_style))
                content.append(Paragraph(checkup.treatment_follow_up.replace('\n', '<br/>'), body_style))
            if checkup:
                content.append(Spacer(1, 0.15*cm))
                reviewer = checkup.doctor.get_full_name() if checkup.doctor else '—'
                content.append(Paragraph(
                    f'<font size=7 color="#94a3b8">Reviewed by {reviewer} on {checkup.created_at.strftime("%d %b %Y")}</font>',
                    body_style))
            content.append(Spacer(1, 0.4*cm))

        # Build
        doc.build(content, onFirstPage=on_page, onLaterPages=on_page)
        buf.seek(0)
        return buf


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationListView(APIView):
    """GET /api/v1/notifications - recent notifications + unread count for the current user."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = Notification.objects.filter(recipient=request.user)
        unread = qs.filter(is_read=False).count()
        items = [{
            "id": n.id,
            "message": n.message,
            "url": n.url,
            "is_read": n.is_read,
            "created_at": n.created_at.isoformat(),
        } for n in qs[:15]]
        return Response({"unread_count": unread, "notifications": items})


class NotificationMarkReadView(APIView):
    """POST /api/v1/notifications/mark-read - mark all (or one) notifications read."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        notif_id = request.data.get('id')
        qs = Notification.objects.filter(recipient=request.user, is_read=False)
        if notif_id:
            qs = qs.filter(id=notif_id)
        updated = qs.update(is_read=True)
        return Response({"marked_read": updated})


# ═══════════════════════════════════════════════════════════════════════════════
# PASSWORD RESET VIEWS (CRUD)
# ═══════════════════════════════════════════════════════════════════════════════

class ForgotPasswordView(View):
    """GET/POST /forgot-password/ - Request password reset via OTP"""
    def get(self, request):
        return render(request, 'forgot_password.html')

    def post(self, request):
        email_or_username = request.POST.get('email_or_username')

        if not email_or_username:
            messages.error(request, 'Please enter your email or username')
            return render(request, 'forgot_password.html')

        # Find user by email or username
        try:
            user = User.objects.get(
                Q(username=email_or_username) | Q(email=email_or_username)
            )
        except User.DoesNotExist:
            # Don't reveal if user exists or not for security
            messages.success(request, 'If an account exists with this email/username, an OTP has been sent.')
            return render(request, 'forgot_password.html')

        # Generate OTP
        otp = ''.join(random.choices(string.digits, k=getattr(settings, 'OTP_LENGTH', 6)))
        expiry_minutes = getattr(settings, 'OTP_EXPIRY_MINUTES', 10)

        # Store OTP in cache with expiration
        cache_key = f'password_reset_otp_{user.username}'
        cache.set(cache_key, {
            'otp': otp,
            'user_id': user.id,
            'email': user.email
        }, expiry_minutes * 60)

        # Send OTP email
        try:
            subject = 'CerviStage AI - Password Reset OTP'
            message = f'''
Dear {user.get_full_name() or user.username},

You have requested to reset your password for CerviStage AI.

Your One-Time Password (OTP) is: {otp}

This OTP will expire in {expiry_minutes} minutes.

If you did not request this password reset, please ignore this email.

Best regards,
CerviStage AI Team
'''

            html_message = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9f9f9; padding: 30px; border: 1px solid #e0e0e0; }}
        .otp {{ background: #fff; padding: 15px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 5px; margin: 20px 0; border: 2px dashed #667eea; border-radius: 5px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 CerviStage AI</h1>
            <p>Password Reset Request</p>
        </div>
        <div class="content">
            <p>Dear <strong>{user.get_full_name() or user.username}</strong>,</p>
            <p>You have requested to reset your password for CerviStage AI.</p>
            <p>Your One-Time Password (OTP) is:</p>
            <div class="otp">{otp}</div>
            <p><strong>This OTP will expire in {expiry_minutes} minutes.</strong></p>
            <p>If you did not request this password reset, please ignore this email.</p>
        </div>
        <div class="footer">
            <p>Best regards,<br>CerviStage AI Team</p>
            <p><em>This is an automated email. Please do not reply.</em></p>
        </div>
    </div>
</body>
</html>
'''

            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                html_message=html_message,
                fail_silently=False
            )

            safe_print(f"[OTP] Sent to {user.email}")

            # Store email in session for verification step
            request.session['reset_email'] = user.email
            request.session['reset_username'] = user.username

            messages.success(request, f'OTP has been sent to your email. Valid for {expiry_minutes} minutes.')
            return redirect('/verify-otp/')

        except Exception as e:
            safe_print(f"[OTP] Failed to send: {e}")
            messages.error(request, 'Failed to send OTP. Please try again later.')
            return render(request, 'forgot_password.html')


class VerifyOTPView(View):
    """GET/POST /verify-otp/ - Verify OTP and reset password"""
    def get(self, request):
        if 'reset_email' not in request.session:
            return redirect('/forgot-password/')
        return render(request, 'verify_otp.html')

    def post(self, request):
        otp = request.POST.get('otp')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        if not otp or not new_password:
            messages.error(request, 'Please fill in all fields')
            return render(request, 'verify_otp.html')

        if new_password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return render(request, 'verify_otp.html')

        if len(new_password) < 8:
            messages.error(request, 'Password must be at least 8 characters long')
            return render(request, 'verify_otp.html')

        # Get user from session
        username = request.session.get('reset_username')
        if not username:
            messages.error(request, 'Session expired. Please start over.')
            return redirect('/forgot-password/')

        # Get user
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            messages.error(request, 'User not found')
            return redirect('/forgot-password/')

        # Get stored OTP from cache
        cache_key = f'password_reset_otp_{username}'
        cached_data = cache.get(cache_key)

        if not cached_data:
            messages.error(request, 'OTP has expired or is invalid. Please request a new OTP.')
            return render(request, 'verify_otp.html')

        # Verify OTP
        if cached_data['otp'] != otp:
            messages.error(request, 'Invalid OTP. Please try again.')
            return render(request, 'verify_otp.html')

        # Reset password
        try:
            user.set_password(new_password)
            user.save()

            # Clear the used OTP and session
            cache.delete(cache_key)
            del request.session['reset_email']
            del request.session['reset_username']

            # Send confirmation email
            try:
                subject = 'CerviStage AI - Password Successfully Reset'
                message = f'''
Dear {user.get_full_name() or user.username},

Your password has been successfully reset for CerviStage AI.

If you did not initiate this change, please contact support immediately.

Best regards,
CerviStage AI Team
'''

                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    [user.email],
                    fail_silently=True
                )
            except:
                pass  # Don't fail if confirmation email fails

            safe_print(f"[PASSWORD RESET] Completed for {user.username}")

            messages.success(request, 'Password reset successfully. You can now login with your new password.')
            return redirect('/login/')

        except Exception as e:
            messages.error(request, f'Failed to reset password: {str(e)}')
            return render(request, 'verify_otp.html')
