from django.http import FileResponse, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.mixins import LoginRequiredMixin
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.db.models import Count, Q
from datetime import datetime, timedelta
from django.utils import timezone
from django.contrib.auth import get_user_model, authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .models import Analysis
from .serializers import AnalysisSerializer, AnalyzeRequestSerializer
from services.inference import InferenceService
import cv2
import numpy as np
import os
import uuid
from django.conf import settings
from PIL import Image

User = get_user_model()



# ─── Template Views ────────────────────────────────────────────────────────────

class HomePageView(View):
    """GET / - Landing page (no authentication required)"""
    def get(self, request):
        # If user is authenticated, redirect to analyze page
        if request.user.is_authenticated:
            return redirect('/analyze/')

        # Show landing page for non-authenticated users
        return render(request, "home.html")


class AnalyzePageView(View):
    """GET /analyze/ - AI analysis tool (requires authentication)"""
    def get(self, request):
        # Require authentication for analyze
        if not request.user.is_authenticated:
            return redirect('/login/?next=/analyze/')

        return render(request, "analyze.html")



class HistoryPageView(View):
    """GET /history/ - Analysis history"""
    def get(self, request):
        # Require authentication for history
        if not request.user.is_authenticated:
            return redirect('/login/?next=/history/')

        # Filter analyses based on user role
        user_role = request.user.profile.role
        if user_role == 'admin':
            # Admins see all analyses
            analyses = Analysis.objects.select_related("created_by", "created_by__profile").order_by("-created_at")[:50]
        else:
            # Users see only their own analyses
            analyses = Analysis.objects.filter(
                created_by=request.user
            ).order_by("-created_at")[:50]

        return render(request, "history.html", {"analyses": analyses, "user_role": user_role})


class LoginPageView(View):
    """GET /login/ - Login page"""
    def get(self, request):
        return render(request, "login.html")


class RegisterPageView(View):
    """GET /register/ - Registration page"""
    def get(self, request):
        return render(request, "register.html")


class DashboardPageView(View):
    """GET /dashboard/ - User dashboard"""
    def get(self, request):
        # Require authentication for dashboard
        if not request.user.is_authenticated:
            return redirect('/login/?next=/dashboard/')

        # If admin, redirect to admin dashboard
        if request.user.profile.role == 'admin':
            return redirect('/admin/')

        # Get user-specific statistics
        total_analyses = Analysis.objects.filter(created_by=request.user).count()

        # Recent analyses (last 7 days)
        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_analyses_count = Analysis.objects.filter(
            created_by=request.user,
            created_at__gte=seven_days_ago
        ).count()

        # Recent activity - show user's analyses
        recent_activity = Analysis.objects.filter(
            created_by=request.user
        ).order_by('-created_at')[:10]

        context = {
            'total_analyses': total_analyses,
            'recent_analyses_count': recent_analyses_count,
            'recent_activity': recent_activity,
        }
        return render(request, "user_dashboard.html", context)


class AnalysisDetailView(View):
    """GET /analysis/<analysis_id> - Detailed analysis results page"""
    def get(self, request, analysis_id):
        analysis = get_object_or_404(Analysis, pk=analysis_id)

        # Convert probabilities and scores to percentages for display
        probabilities_pct = {k: round(float(v) * 100, 2) for k, v in analysis.probabilities.items()}
        confidence_pct = round(analysis.confidence * 100, 1)
        uncertainty_pct = round(analysis.uncertainty * 100, 1)

        context = {
            'analysis': analysis,
            'probabilities': probabilities_pct,
            'confidence_pct': confidence_pct,
            'uncertainty_pct': uncertainty_pct,
        }
        return render(request, "analysis_detail.html", context)


class SettingsPageView(View):
    """GET /settings/ - User settings page"""
    def get(self, request):
        # Require authentication for settings
        if not request.user.is_authenticated:
            return redirect('/login/?next=/settings/')

        if request.user.profile.role == 'admin':
            return render(request, "admin_settings.html")
        return render(request, "settings.html")


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
        return render(request, "user_profile.html", context)


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


# Rename the API detail view to avoid conflict
class AnalysisApiDetailView(APIView):
    """GET /v1/analyze/<analysis_id> - API endpoint for analysis details"""
    def get(self, request, analysis_id):
        try:
            analysis = Analysis.objects.get(pk=analysis_id)
        except Analysis.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(AnalysisSerializer(analysis).data)



# ─── Authentication Views (API) ────────────────────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
class RegisterView(APIView):
    """POST /v1/auth/register - Register a new user (always as 'user' role)"""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def post(self, request):
        username = request.data.get("username")
        email = request.data.get("email")
        password = request.data.get("password")
        full_name = request.data.get("full_name", "")

        # Validate required fields
        if not username or not email or not password:
            return Response(
                {"error": "Username, email, and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if user already exists
        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "Username already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(email=email).exists():
            return Response(
                {"error": "Email already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Create user with 'user' role (regular user)
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=full_name.split()[0] if full_name else "",
            last_name=" ".join(full_name.split()[1:]) if full_name else "",
        )
        # Set role to 'user' for all registered users
        # Admin users must be created via Django superuser command
        user.profile.role = "user"
        user.profile.save()

        # Generate tokens
        refresh = RefreshToken.for_user(user)

        return Response({
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.get_full_name(),
                "role": user.profile.role,
            },
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)


@method_decorator(csrf_exempt, name='dispatch')
class LoginView(APIView):
    """POST /v1/auth/login - Login user and create session"""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        if not username or not password:
            return Response(
                {"error": "Username and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

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

        # Redirect based on role
        if user.profile.role == 'admin':
            redirect_url = '/admin/'
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

class AdminAnalysisDetailView(View):
    """GET /admin/analysis/<analysis_id> - Admin view for detailed analysis results"""
    def get(self, request, analysis_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/analysis/' + str(analysis_id))

        if request.user.profile.role != 'admin':
            return redirect('/admin/')

        analysis = get_object_or_404(Analysis, pk=analysis_id)

        probabilities_pct = {k: round(float(v) * 100, 2) for k, v in analysis.probabilities.items()}
        confidence_pct = round(analysis.confidence * 100, 1)
        uncertainty_pct = round(analysis.uncertainty * 100, 1)

        context = {
            'analysis': analysis,
            'probabilities': probabilities_pct,
            'confidence_pct': confidence_pct,
            'uncertainty_pct': uncertainty_pct,
            'user_role': 'admin',
        }
        return render(request, "admin_analysis_detail.html", context)


class AdminDashboardView(View):
    """GET /admin/ - Dashboard for all users (admin and users)"""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/')

        user_role = request.user.profile.role

        if user_role == 'admin':
            total_users = User.objects.count()
            total_analyses = Analysis.objects.count()

            thirty_days_ago = timezone.now() - timedelta(days=30)
            recent_users = User.objects.filter(
                date_joined__gte=thirty_days_ago
            ).count()

            role_dist = User.objects.values('profile__role').annotate(
                count=Count('id')
            ).order_by('-count')

            recent_analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile'
            ).order_by('-created_at')[:20]

            recent_user_registrations = User.objects.select_related('profile').exclude(
                profile__role='admin'
            ).order_by('-date_joined')[:10]

            context = {
                'total_users': total_users,
                'total_analyses': total_analyses,
                'recent_users': recent_users,
                'role_dist': list(role_dist),
                'recent_analyses': recent_analyses,
                'recent_user_registrations': recent_user_registrations,
                'user_role': 'admin',
            }
        else:
            total_analyses = Analysis.objects.filter(
                created_by=request.user
            ).count()

            recent_analyses = Analysis.objects.filter(
                created_by=request.user
            ).order_by('-created_at')[:20]

            context = {
                'total_analyses': total_analyses,
                'recent_analyses': recent_analyses,
                'user_role': 'user',
            }

        return render(request, "admin_dashboard.html", context)


class AdminUsersView(View):
    """GET /admin/users - View all users (admin only)"""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/users')

        if request.user.profile.role != 'admin':
            return redirect('/admin/')

        users = User.objects.select_related('profile').exclude(
            profile__role='admin'
        ).order_by('-date_joined')
        context = {
            'users': users,
        }
        return render(request, "admin_users.html", context)


class AdminUserHistoryView(View):
    """GET /admin/user/<user_id>/history - View specific user's history (admin only)"""
    def get(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/user/' + str(user_id) + '/history')

        if request.user.profile.role != 'admin':
            return redirect('/admin/')

        target_user = get_object_or_404(User, pk=user_id)

        analyses = Analysis.objects.filter(
            created_by=target_user
        ).order_by('-created_at')[:50]

        total_analyses = analyses.count()
        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses
        else:
            avg_confidence = 0

        context = {
            'target_user': target_user,
            'analyses': analyses,
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence * 100, 2),
        }
        return render(request, "admin_user_history.html", context)


class AdminRegisteredUsersView(View):
    """GET /admin/registered-users - View and manage all registered users (admin only)"""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/registered-users')

        if request.user.profile.role != 'admin':
            return redirect('/admin/')

        users = User.objects.select_related('profile').exclude(
            profile__role='admin'
        ).order_by('-date_joined')

        total_users = users.count()

        from django.utils import timezone
        now = timezone.now()
        first_day_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        new_users_month = users.filter(date_joined__gte=first_day_of_month).count()

        active_users = users.filter(
            profile__role='user',
            is_active=True
        ).count()

        from django.core.paginator import Paginator
        paginator = Paginator(users, 25)
        page_number = request.GET.get('page')
        users_page = paginator.get_page(page_number)

        context = {
            'users': users_page,
            'total_users': total_users,
            'new_users_month': new_users_month,
            'active_users': active_users,
        }
        return render(request, "admin_registered_users.html", context)


class AdminTestResultsView(View):
    """GET /admin/test-results - View all test results"""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/test-results')

        user_role = request.user.profile.role

        if user_role == 'admin':
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile'
            ).all().order_by('-created_at')
        else:
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile'
            ).filter(created_by=request.user).order_by('-created_at')

        total_analyses = analyses.count()

        severity_counts = {}
        for analysis in analyses:
            label = analysis.predicted_label
            severity_counts[label] = severity_counts.get(label, 0) + 1

        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_analyses = analyses.filter(created_at__gte=seven_days_ago).count()

        from django.core.paginator import Paginator
        paginator = Paginator(analyses, 50)
        page_number = request.GET.get('page')
        analyses_page = paginator.get_page(page_number)

        context = {
            'analyses': analyses_page,
            'total_analyses': total_analyses,
            'severity_counts': severity_counts,
            'recent_analyses': recent_analyses,
            'user_role': user_role,
        }
        return render(request, "admin_test_results.html", context)


class AdminTestHistoryView(View):
    """GET /admin/test-history - View detailed test history"""
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/test-history')

        user_role = request.user.profile.role

        if user_role == 'admin':
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile'
            ).all().order_by('-created_at')
        else:
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile'
            ).filter(created_by=request.user).order_by('-created_at')

        total_analyses = analyses.count()

        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses
        else:
            avg_confidence = 0

        from django.core.paginator import Paginator
        paginator = Paginator(analyses, 50)
        page_number = request.GET.get('page')
        analyses_page = paginator.get_page(page_number)

        context = {
            'analyses': analyses_page,
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence * 100, 2),
            'user_role': user_role,
        }
        return render(request, "admin_test_history.html", context)


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
    print(f"[DEBUG] Generating frames in: {anim_dir_path}")
    
    width, height = 640, 640
    frame_urls = []
    
    for i, (record, pred) in enumerate(zip(analysis_records, predictions_list)):
        try:
            img_path = record.image.path
            print(f"[DEBUG] Frame {i}: Reading image from {img_path}")
            if not os.path.exists(img_path):
                print(f"[WARN] Image file not found: {img_path}")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to read image with cv2: {img_path}")
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
            print(f"[DEBUG] Frame {i}: Saved to {frame_filepath}")
            
            frame_url = f"{settings.MEDIA_URL}animations/{anim_dir_name}/{frame_filename}"
            frame_urls.append(frame_url)
            print(f"[DEBUG] Frame {i}: URL = {frame_url}")
            
        except Exception as e:
            print(f"[ERROR] Frame {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[DEBUG] Generated {len(frame_urls)} frames total")
    return frame_urls


class AnalyzeView(APIView):
    """POST /v1/analyze — upload multiple images, run AI staging on each, return averaged result."""
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Get all files from request (supports both single and multiple uploads)
        files = request.FILES.getlist('file')
        MAX_FILES = 200

        if len(files) > MAX_FILES:
            return Response(
                {
                    "error": f"Maximum {MAX_FILES} images allowed"
                },
                status=400
            )
        if not files:
            return Response(
                {"error": "No image files provided"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if heatmap generation should be skipped (for batch processing performance)
        skip_heatmap = request.query_params.get('skip_heatmap', '').lower() in ('true', '1', 'yes')
        if skip_heatmap:
            print("[DEBUG] Heatmap generation DISABLED for faster batch processing")

        # Log for debugging
        print(f"[DEBUG] Received {len(files)} files for analysis")

        allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
        svc = InferenceService.get()
        
        if svc.session is None:
            return Response(
                {"error": "Model not loaded. Copy cervistage_net.onnx to the models/ directory."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        predictions_list = []
        analysis_records = []

        # Process each uploaded image
        for idx, image_file in enumerate(files):
            # Validate MIME type
            if image_file.content_type not in allowed_types:
                return Response(
                    {"error": f"File {idx+1} ({image_file.name}): Unsupported type {image_file.content_type}. Allowed: JPEG, PNG, BMP, TIFF"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                print(f"[DEBUG] Processing file {idx+1}/{len(files)}: {image_file.name}")
                prediction = svc.predict(
                    image_file,
                    skip_heatmap=skip_heatmap
                )
                predictions_list.append(prediction)

                # Store individual analysis record in database
                image_file.seek(0)
                extra_fields = {}
                for key in ("stage_label", "risk_level", "risk_color", "explanation", "confidence_interpretation"):
                    extra_fields[key] = prediction.pop(key)

                image_file.seek(0)
                analysis = Analysis.objects.create(
                    created_by=request.user,
                    image=image_file,
                    **prediction,
                )
                analysis_records.append(analysis)
                print(f"[DEBUG] ✓ Stored analysis for file {idx+1}: {analysis.analysis_id}")

            except RuntimeError as e:
                return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Average predictions across all images
        print(f"[DEBUG] Averaging {len(predictions_list)} predictions")
        averaged_prediction = svc.average_predictions(predictions_list)
        

        # Generate frames of the results
        frame_urls = []
        if len(files) <= 30:
            frame_urls = generate_results_frames(
                analysis_records,
                predictions_list
    )

        # Build response with averaged results
        # Build individual results with their classifications
        individual_results = []
        for idx, (pred, record) in enumerate(zip(predictions_list, analysis_records)):
            individual_results.append({
                "analysis_id": str(record.analysis_id),
                "predicted_class": pred["predicted_class"],
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"],
                "image_url": record.image.url if record.image else None,
            })

        resp = {
            "predicted_class": averaged_prediction["predicted_class"],
            "predicted_label": averaged_prediction["predicted_label"],
            "stage_label": averaged_prediction["stage_label"],
            "probabilities": averaged_prediction["probabilities"],
            "confidence": averaged_prediction["confidence"],
            "uncertainty": averaged_prediction["uncertainty"],
            "confidence_level": averaged_prediction["confidence_level"],
            "risk_level": averaged_prediction["risk_level"],
            "risk_color": averaged_prediction["risk_color"],
            "confidence_interpretation": averaged_prediction["confidence_interpretation"],
            "recommendation": averaged_prediction["recommendation"],
            "explanation": averaged_prediction["explanation"],
            "images_analyzed": len(files),
            "analysis_id": str(analysis_records[0].analysis_id) if analysis_records else None,
            "individual_analyses": [str(a.analysis_id) for a in analysis_records],
            "individual_results": individual_results,
            "frame_urls": frame_urls,
        }

        return Response(resp, status=status.HTTP_201_CREATED)


class HealthView(APIView):
    """GET /v1/health"""

    def get(self, request):
        model_loaded = InferenceService.get().session is not None
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

        response = HttpResponse(pdf_buffer, content_type="application/pdf")
        response["Content-Disposition"] = (
            f'attachment; filename="cervistage_report_{str(analysis_id)[:8]}.pdf"'
        )
        return response

    def _generate_pdf(self, analysis, user_name):
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4

        # Header band
        c.setFillColor(colors.HexColor("#1e3a8a"))
        c.rect(0, h - 4 * cm, w, 4 * cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(2 * cm, h - 2 * cm, "CerviStage AI — Clinical Report")
        c.setFont("Helvetica", 11)
        c.drawString(2 * cm, h - 3 * cm, "AI-Based Cervical Cancer Detection & Staging System")

        # User info
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(2 * cm, h - 6 * cm, "Report Information")
        c.setFont("Helvetica", 11)
        if user_name:
            c.drawString(2 * cm, h - 7 * cm, f"Analyzed by: {user_name}")
        c.drawString(2 * cm, h - 7.8 * cm, f"Date: {analysis.created_at.strftime('%Y-%m-%d %H:%M')} UTC")

        # Staging result
        stage_colors = {0: "#22c55e", 1: "#84cc16", 2: "#eab308", 3: "#f97316", 4: "#ef4444"}
        color_hex = stage_colors.get(analysis.predicted_class, "#64748b")

        c.setFont("Helvetica-Bold", 13)
        c.drawString(2 * cm, h - 11 * cm, "Staging Result")

        c.setFillColor(colors.HexColor(color_hex))
        c.setFont("Helvetica-Bold", 22)
        c.drawString(2 * cm, h - 12.5 * cm, analysis.predicted_label)

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 11)
        c.drawString(2 * cm, h - 13.5 * cm, f"Confidence: {round(analysis.confidence * 100)}%  |  "
                     f"Uncertainty: {round(analysis.uncertainty * 100)}%  |  "
                     f"AI Level: {analysis.confidence_level}")

        # Probabilities
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, h - 15 * cm, "Class Probabilities")
        y = h - 16 * cm
        for label, prob in analysis.probabilities.items():
            c.setFont("Helvetica", 10)
            c.drawString(2 * cm, y, f"{label}:")
            bar_w = float(prob) * 8 * cm
            c.setFillColor(colors.HexColor("#3b82f6"))
            c.rect(6 * cm, y - 0.1 * cm, bar_w, 0.35 * cm, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.drawString(6 * cm + bar_w + 0.2 * cm, y, f"{round(float(prob) * 100)}%")
            y -= 0.7 * cm

        # Recommendation
        c.setFillColor(colors.HexColor("#fef3c7"))
        c.rect(2 * cm, y - 1.5 * cm, w - 4 * cm, 1.8 * cm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#92400e"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2.3 * cm, y - 0.5 * cm, "Clinical Recommendation")
        c.setFont("Helvetica", 10)
        c.drawString(2.3 * cm, y - 1.2 * cm, analysis.recommendation[:100])

        # Footer
        c.setFillColor(colors.gray)
        c.setFont("Helvetica", 8)
        c.drawString(2 * cm, 1.5 * cm, "Generated by CerviStage AI — For research and screening assistance only.")
        c.drawString(2 * cm, 0.8 * cm, f"Analysis ID: {analysis.analysis_id}")

        c.save()
        buf.seek(0)
        return buf
