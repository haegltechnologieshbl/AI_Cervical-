"""Role-specific views — shared imports/helpers come from core.views."""
from .views import *  # noqa: F401,F403


class AdminAnalysisDetailView(RoleRequiredMixin, View):
    """GET /admin/analysis/<analysis_id> - Admin view for detailed analysis results"""
    allowed_roles = ('admin',)
    def get(self, request, analysis_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/analysis/' + str(analysis_id))

        if request.user.profile.role != 'admin':
            return redirect('/admin/')

        analysis = get_object_or_404(Analysis, pk=analysis_id)

        # If this is part of a batch but not the primary analysis, redirect to primary —
        # unless ?single=1 is passed, which lets the admin inspect one image on its own.
        view_single = request.GET.get('single') == '1'
        if analysis.batch_id and not analysis.is_batch_primary and not view_single:
            try:
                primary_analysis = Analysis.objects.filter(batch_id=analysis.batch_id, is_batch_primary=True).first()
                if primary_analysis:
                    return redirect(f'/admin/analysis/{primary_analysis.analysis_id}')
            except:
                pass  # If primary not found, show current analysis

        # When inspecting a single image from a batch, expose the link back to the batch
        batch_primary_id = None
        if analysis.batch_id and not analysis.is_batch_primary:
            primary = Analysis.objects.filter(batch_id=analysis.batch_id, is_batch_primary=True).first()
            batch_primary_id = primary.analysis_id if primary else None

        probabilities_pct = {k: round(float(v) * 100, 2) for k, v in analysis.probabilities.items()}
        confidence_pct = round(analysis.confidence * 100, 1)

        # Fetch batch information if this is a batch primary analysis
        batch_analyses = []
        batch_data = None
        if analysis.batch_id and analysis.is_batch_primary:
            batch_analyses = list(Analysis.objects.filter(batch_id=analysis.batch_id).order_by('batch_position'))
            batch_data = analysis.batch_aggregated_data
            # Add confidence percentage and probabilities to each batch analysis
            for ba in batch_analyses:
                ba.confidence_pct = round(ba.confidence * 100, 1)
                ba.probabilities_pct = {k: round(v * 100, 1) for k, v in ba.probabilities.items()}

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
            'user_role': 'admin',
            'batch_analyses': batch_analyses,
            'batch_data': batch_data,
            'is_batch': bool(analysis.batch_id),
            'pap_smear': pap_smear,
            'checkup': checkup,
            'doctor_checkups': doctor_checkups,
            'batch_primary_id': batch_primary_id,
        }
        return render(request, "superuser/admin_analysis_detail.html", context)


class AdminDashboardView(RoleRequiredMixin, View):
    """GET /admin/ - Dashboard for all users (admin and users)"""
    allowed_roles = ('admin',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/')

        user_role = request.user.profile.role
        is_admin = user_role == 'admin' or request.user.is_superuser

        if is_admin:
            user_role = 'admin'
            total_users = User.objects.count()
            total_analyses = Analysis.objects.count()

            thirty_days_ago = timezone.now() - timedelta(days=30)
            recent_users = User.objects.filter(
                date_joined__gte=thirty_days_ago
            ).count()

            role_dist = User.objects.values('profile__role').annotate(
                count=Count('id')
            ).order_by('-count')

            recent_analyses = Analysis.objects.filter(
                Q(batch_id__isnull=True) | Q(is_batch_primary=True)
            ).select_related(
                'created_by', 'created_by__profile', 'patient'
            ).order_by('-created_at')[:20]

            # Separate recent registrations by role
            recent_doctor_registrations = User.objects.select_related('profile').filter(
                profile__role='user',
                is_superuser=False
            ).order_by('-date_joined')[:10]

            recent_technician_registrations = User.objects.select_related('profile').filter(
                profile__role='senior_technician',
                is_superuser=False
            ).order_by('-date_joined')[:10]

            # Count stats
            total_doctors = User.objects.filter(
                profile__role='user',
                is_superuser=False
            ).count()

            total_technicians = User.objects.filter(
                profile__role='senior_technician',
                is_superuser=False
            ).count()

            total_patients_registered = Patient.objects.count()

            # New this month
            month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            new_doctors_month = User.objects.filter(
                profile__role='user',
                is_superuser=False,
                date_joined__gte=month_start
            ).count()

            new_technicians_month = User.objects.filter(
                profile__role='senior_technician',
                is_superuser=False,
                date_joined__gte=month_start
            ).count()

            new_patients_month = Patient.objects.filter(
                created_at__gte=month_start
            ).count()

            # Get recent patients for dashboard
            recent_patients = Patient.objects.select_related(
                'created_by', 'created_by__profile'
            ).order_by('-created_at')[:10]

            context = {
                'total_users': total_users,
                'total_analyses': total_analyses,
                'recent_users': recent_users,
                'role_dist': list(role_dist),
                'recent_analyses': recent_analyses,
                'recent_doctor_registrations': recent_doctor_registrations,
                'recent_technician_registrations': recent_technician_registrations,
                'recent_patients': recent_patients,
                'total_doctors': total_doctors,
                'total_technicians': total_technicians,
                'total_patients_registered': total_patients_registered,
                'new_doctors_month': new_doctors_month,
                'new_technicians_month': new_technicians_month,
                'new_patients_month': new_patients_month,
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

        return render(request, "superuser/admin_dashboard.html", context)


from django.core.paginator import Paginator
from django.contrib import messages

class AdminDoctorsView(RoleRequiredMixin, View):
    """GET /admin/doctors - View all doctors"""
    allowed_roles = ('admin',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/doctors/')

        users_list = User.objects.select_related('profile').filter(
            profile__role='user',
            is_superuser=False
        ).order_by('-date_joined')

        paginator = Paginator(users_list, 10)
        users = paginator.get_page(request.GET.get('page'))

        return render(request, "superuser/admin_doctors.html", {'users': users})

    def post(self, request):
        """Handle delete user"""
        if not request.user.is_authenticated:
            return redirect('/login/')

        action = request.POST.get('action')
        user_id = request.POST.get('user_id')

        try:
            user = User.objects.get(id=user_id, is_superuser=False)
        except User.DoesNotExist:
            messages.error(request, 'User not found.')
            return redirect('/admin/doctors/')

        if action == 'delete':
            username = user.username
            user.delete()
            messages.success(request, f'User "{username}" deleted successfully.')

        return redirect('/admin/doctors/')

class AddDoctorView(RoleRequiredMixin, View):
    """GET /admin/doctor/add/ - Add a new doctor account"""
    allowed_roles = ('admin',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/doctor/add')

        # Only admins can add doctors
        # With this:
        if not request.user.is_superuser and request.user.profile.role != 'admin':
            messages.error(request, 'Only administrators can add doctors.')
            return redirect('/admin/doctors/')

        return render(request, 'superuser/add_doctor.html')

    def post(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/')

        # Only admins can add doctors
        if not request.user.is_superuser and request.user.profile.role != 'admin':
            messages.error(request, 'Only administrators can add doctors.')
            return redirect('/admin/doctors/')

        username = request.POST.get('username')
        email = request.POST.get('email')
        first_name = request.POST.get('first_name', '')
        last_name = request.POST.get('last_name', '')
        password = request.POST.get('password')
        phone = request.POST.get('phone', '')
        department = request.POST.get('department', '')
        license_number = request.POST.get('license_number', '')
        role = request.POST.get('role', 'user')

        # Validation
        if not username or not email or not password:
            messages.error(request, 'Username, email, and password are required.')
            return render(request, 'superuser/add_doctor.html', {
                'username': username,
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'phone': phone,
                'department': department,
                'license_number': license_number,
                'role': role,
            })

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return render(request, 'superuser/add_doctor.html', {
                'username': username,
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'phone': phone,
                'department': department,
                'license_number': license_number,
                'role': role,
            })

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists.')
            return render(request, 'superuser/add_doctor.html', {
                'username': username,
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'phone': phone,
                'department': department,
                'license_number': license_number,
                'role': role,
            })

        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )

        # Update profile
        user.profile.role = role
        user.profile.phone = phone
        user.profile.department = department
        user.profile.license_number = license_number
        user.profile.save()

        messages.success(request, f'Doctor "{username}" has been created successfully.')
        return redirect('/admin/doctors/')

class EditDoctorView(RoleRequiredMixin, View):
    """GET /admin/user/<id>/edit/ - Edit user"""
    allowed_roles = ('admin',)
    def get(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/')

        try:
            edit_user = User.objects.select_related('profile').get(id=user_id, is_superuser=False)
        except User.DoesNotExist:
            messages.error(request, 'User not found.')
            return redirect('/admin/doctors/')

        return render(request, 'superuser/admin_edit_doctors.html', {'edit_user': edit_user})

    def post(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/')

        try:
            edit_user = User.objects.select_related('profile').get(id=user_id, is_superuser=False)
        except User.DoesNotExist:
            messages.error(request, 'User not found.')
            return redirect('/admin/doctors/')

        # Update User fields
        edit_user.first_name = request.POST.get('first_name', edit_user.first_name)
        edit_user.last_name = request.POST.get('last_name', edit_user.last_name)
        edit_user.email = request.POST.get('email', edit_user.email)
        edit_user.save()

        # Update Profile fields
        edit_user.profile.role = request.POST.get('role', edit_user.profile.role)
        edit_user.profile.phone = request.POST.get('phone') or edit_user.profile.phone
        edit_user.profile.department = request.POST.get('department') or edit_user.profile.department
        edit_user.profile.license_number = request.POST.get('license_number') or edit_user.profile.license_number
        edit_user.profile.save()

        # Handle password change
        new_password = request.POST.get('password')
        if new_password:
            edit_user.set_password(new_password)
            edit_user.save()

        messages.success(request, f'User "{edit_user.username}" updated successfully!')
        return redirect('/admin/doctors/')


class AdminTechniciansView(RoleRequiredMixin, View):
    """GET /admin/technicians/ - List all senior technicians (POST to delete)"""
    allowed_roles = ('admin',)

    def _admin_check(self, request, redirect_to='/admin/technicians/'):
        if not request.user.is_authenticated:
            return redirect('/login/')
        if request.user.profile.role != 'admin' and not request.user.is_superuser:
            messages.error(request, 'Access denied.')
            return redirect('/admin/')
        return None

    def get(self, request):
        guard = self._admin_check(request)
        if guard:
            return guard

        technicians_list = User.objects.select_related('profile').filter(
            profile__role='senior_technician',
            is_superuser=False
        ).order_by('-date_joined')

        paginator = Paginator(technicians_list, 10)
        technicians = paginator.get_page(request.GET.get('page'))

        return render(request, 'superuser/admin_technicians.html', {'technicians': technicians})

    def post(self, request):
        guard = self._admin_check(request)
        if guard:
            return guard

        action = request.POST.get('action')
        user_id = request.POST.get('user_id')

        try:
            user = User.objects.get(id=user_id, profile__role='senior_technician', is_superuser=False)
        except User.DoesNotExist:
            messages.error(request, 'Technician not found.')
            return redirect('/admin/technicians/')

        if action == 'delete':
            name = user.get_full_name() or user.username
            user.delete()
            messages.success(request, f'Technician "{name}" deleted successfully.')

        return redirect('/admin/technicians/')


class AddTechnicianView(RoleRequiredMixin, View):
    """GET/POST /admin/technician/add/ — Create a senior technician (username = email)"""
    allowed_roles = ('admin',)

    def _admin_check(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/technician/add/')
        if request.user.profile.role != 'admin' and not request.user.is_superuser:
            messages.error(request, 'Only administrators can add technicians.')
            return redirect('/admin/technicians/')
        return None

    def get(self, request):
        guard = self._admin_check(request)
        if guard:
            return guard
        return render(request, 'superuser/admin_add_technician.html')

    def post(self, request):
        guard = self._admin_check(request)
        if guard:
            return guard

        email = request.POST.get('email', '').strip().lower()
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        password = request.POST.get('password', '')
        phone = request.POST.get('phone', '').strip()
        department = request.POST.get('department', '').strip()
        employee_id = request.POST.get('employee_id', '').strip()

        ctx = {
            'email': email, 'first_name': first_name, 'last_name': last_name,
            'phone': phone, 'department': department, 'employee_id': employee_id,
        }

        if not email or not password:
            messages.error(request, 'Email and password are required.')
            return render(request, 'superuser/admin_add_technician.html', ctx)

        if User.objects.filter(username=email).exists() or User.objects.filter(email=email).exists():
            messages.error(request, 'A user with this email already exists.')
            return render(request, 'superuser/admin_add_technician.html', ctx)

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )
        user.profile.role = 'senior_technician'
        user.profile.phone = phone
        user.profile.department = department
        user.profile.license_number = employee_id
        user.profile.save()

        messages.success(request, f'Technician "{email}" created successfully.')
        return redirect('/admin/technicians/')


class EditTechnicianView(RoleRequiredMixin, View):
    """GET/POST /admin/technician/<id>/edit/ — Edit a senior technician"""
    allowed_roles = ('admin',)

    def _get_technician(self, user_id):
        try:
            return User.objects.select_related('profile').get(
                id=user_id, profile__role='senior_technician', is_superuser=False
            )
        except User.DoesNotExist:
            return None

    def get(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/')
        tech = self._get_technician(user_id)
        if not tech:
            messages.error(request, 'Technician not found.')
            return redirect('/admin/technicians/')
        return render(request, 'superuser/admin_edit_technician.html', {'tech': tech})

    def post(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/')
        tech = self._get_technician(user_id)
        if not tech:
            messages.error(request, 'Technician not found.')
            return redirect('/admin/technicians/')

        email = request.POST.get('email', '').strip().lower()
        if email and email != tech.email:
            if User.objects.filter(email=email).exclude(id=tech.id).exists():
                messages.error(request, 'Email already in use.')
                return render(request, 'superuser/admin_edit_technician.html', {'tech': tech})
            tech.email = email
            tech.username = email

        tech.first_name = request.POST.get('first_name', tech.first_name).strip()
        tech.last_name = request.POST.get('last_name', tech.last_name).strip()
        tech.save()

        tech.profile.phone = request.POST.get('phone', tech.profile.phone).strip()
        tech.profile.department = request.POST.get('department', tech.profile.department).strip()
        tech.profile.license_number = request.POST.get('employee_id', tech.profile.license_number).strip()
        tech.profile.save()

        new_password = request.POST.get('password', '').strip()
        if new_password:
            tech.set_password(new_password)
            tech.save()

        messages.success(request, f'Technician "{tech.email}" updated successfully.')
        return redirect('/admin/technicians/')


class AdminDoctorsDetailView(RoleRequiredMixin, View):
    """GET /admin/doctors/<user_id>/ - View detailed doctor information"""
    allowed_roles = ('admin',)
    def get(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/doctors/' + str(user_id))

        # Only allow admins to view doctor details
        if not request.user.is_superuser and request.user.profile.role != 'admin':
            return redirect('/admin/')

        target_user = get_object_or_404(User, pk=user_id)

        # Count total unique patients tested by this doctor (via analyses)
        # Do this BEFORE slicing to get accurate total count
        total_tested_patients = Patient.objects.filter(
            analyses__created_by=target_user
        ).distinct().count()

        # Get unique patients tested by this doctor
        tested_patients = Patient.objects.filter(
            analyses__created_by=target_user
        ).distinct().order_by('-analyses__created_at')

        from django.core.paginator import Paginator
        patients_paginator = Paginator(tested_patients, 10)
        patients_page = patients_paginator.get_page(request.GET.get('page'))

        # Get user's analyses
        analyses = Analysis.objects.filter(created_by=target_user).order_by('-created_at')
        total_analyses = analyses.count()

        # Calculate statistics
        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses
            latest_analysis = analyses.first()
        else:
            avg_confidence = 0
            latest_analysis = None

        # Severity distribution
        severity_counts = {}
        for analysis in analyses:
            label = analysis.predicted_label
            severity_counts[label] = severity_counts.get(label, 0) + 1

        # Recent activity
        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_analyses = analyses.filter(created_at__gte=seven_days_ago).count()

        context = {
            'target_user': target_user,
            'patients': patients_page,
            'total_patients': total_tested_patients,
            'analyses': analyses[:10],
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence * 100, 2),
            'latest_analysis': latest_analysis,
            'severity_counts': severity_counts,
            'recent_analyses': recent_analyses,
        }
        return render(request, "superuser/admin_doctors_detail.html", context)


class AdminUserHistoryView(RoleRequiredMixin, View):
    """GET /admin/user/<user_id>/history - View specific user's history (admin only)"""
    allowed_roles = ('admin',)
    def get(self, request, user_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/user/' + str(user_id) + '/history')

        # Allow access if user is admin OR user (authenticated users can see the list)
        # Comment out the strict admin check for now
        # if request.user.profile.role != 'admin':
        #     return redirect('/admin/')

        target_user = get_object_or_404(User, pk=user_id)

        analyses = Analysis.objects.filter(
            created_by=target_user
        ).filter(
            Q(batch_id__isnull=True) | Q(is_batch_primary=True)
        ).order_by('-created_at')

        total_analyses = analyses.count()
        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses
        else:
            avg_confidence = 0

        from django.core.paginator import Paginator
        paginator = Paginator(analyses, 10)
        analyses_page = paginator.get_page(request.GET.get('page'))

        context = {
            'target_user': target_user,
            'analyses': analyses_page,
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence * 100, 2),
        }
        return render(request, "superuser/admin_user_history.html", context)


class AdminPatientsView(RoleRequiredMixin, View):
    """GET /admin/patients - View and manage all patients created by doctors (admin only)"""
    allowed_roles = ('admin',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/patients')

        # Show patients created by doctors (from Patient model, not User model)
        patients = Patient.objects.select_related(
            'created_by', 'created_by__profile'
        ).order_by('-created_at')

        # Get filter parameters
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')

        # Apply date filters if provided
        if date_from:
            from datetime import datetime
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
                # Make it timezone-aware
                from django.utils import timezone
                date_from_obj = timezone.make_aware(date_from_obj)
                patients = patients.filter(created_at__gte=date_from_obj)
            except ValueError:
                pass

        if date_to:
            from datetime import datetime, timedelta
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
                # Include the entire day (add 1 day)
                from django.utils import timezone
                date_to_obj = timezone.make_aware(date_to_obj) + timedelta(days=1)
                patients = patients.filter(created_at__lt=date_to_obj)
            except ValueError:
                pass

        total_patients = patients.count()

        from django.utils import timezone
        now = timezone.now()
        first_day_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        new_patients_month = patients.filter(created_at__gte=first_day_of_month).count()

        from django.core.paginator import Paginator
        paginator = Paginator(patients, 10)
        page_number = request.GET.get('page')
        patients_page = paginator.get_page(page_number)

        context = {
            'patients': patients_page,
            'total_users': total_patients,  # Keep same template variable name
            'new_users_month': new_patients_month,  # Keep same template variable name
            'active_users': Patient.objects.count(),  # Keep same template variable name
            'date_from': date_from,
            'date_to': date_to,
        }
        return render(request, "superuser/admin_registered_users.html", context)


class AdminTestResultsView(RoleRequiredMixin, View):
    """GET /admin/test-results - View all test results"""
    allowed_roles = ('admin',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/test-results')

        user_role = request.user.profile.role

        if user_role == 'admin':
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile', 'patient'
            ).all()
        else:
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile', 'patient'
            ).filter(created_by=request.user)

        # Filter out non-primary batch analyses
        analyses = analyses.filter(
            Q(batch_id__isnull=True) | Q(is_batch_primary=True)
        ).order_by('-created_at')

        # Global counts for tab badges (always unfiltered)
        doctor_count = analyses.filter(created_by__profile__role='user').count()
        technician_count = analyses.filter(created_by__profile__role='senior_technician').count()
        total_all = analyses.count()

        # Role filter (doctor / senior_technician / all)
        role_filter = request.GET.get('role', '')
        if role_filter in ('user', 'senior_technician'):
            analyses = analyses.filter(created_by__profile__role=role_filter)

        total_analyses = total_all

        severity_counts = {}
        for analysis in analyses:
            label = analysis.predicted_label
            severity_counts[label] = severity_counts.get(label, 0) + 1

        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_analyses = analyses.filter(created_at__gte=seven_days_ago).count()

        from django.core.paginator import Paginator
        paginator = Paginator(analyses, 10)
        page_number = request.GET.get('page')
        analyses_page = paginator.get_page(page_number)

        # Latest doctor checkup recommendation per patient on this page
        from .models import CheckupRecommendation
        page_patient_ids = {a.patient_id for a in analyses_page if a.patient_id}
        latest_checkups = {}
        if page_patient_ids:
            for rec in CheckupRecommendation.objects.filter(
                patient_id__in=page_patient_ids
            ).order_by('patient_id', '-created_at'):
                latest_checkups.setdefault(rec.patient_id, rec)

        analyses_with_pct = []
        for analysis in analyses_page:
            rec = latest_checkups.get(analysis.patient_id)
            analyses_with_pct.append({
                'analysis': analysis,
                'confidence_pct': round(analysis.confidence * 100, 1),
                'doctor_checkups': rec.recommended_list() if rec else [],
            })

        context = {
            'analyses': analyses_page,
            'analyses_with_pct': analyses_with_pct,
            'total_analyses': total_analyses,
            'total_all': total_all,
            'doctor_count': doctor_count,
            'technician_count': technician_count,
            'severity_counts': severity_counts,
            'recent_analyses': recent_analyses,
            'user_role': user_role,
            'role_filter': role_filter,
        }
        return render(request, "superuser/admin_test_results.html", context)


class AdminTestHistoryView(RoleRequiredMixin, View):
    """GET /admin/test-history - View detailed test history"""
    allowed_roles = ('admin',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/test-history')

        user_role = request.user.profile.role

        if user_role == 'admin':
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile', 'patient'
            ).all()
        else:
            analyses = Analysis.objects.select_related(
                'created_by', 'created_by__profile', 'patient'
            ).filter(created_by=request.user)

        # Filter out non-primary batch analyses (show only the primary analysis from each batch)
        # This prevents showing 10 separate entries when user uploads 10 images as a batch
        analyses = analyses.filter(
            Q(batch_id__isnull=True) | Q(is_batch_primary=True)
        ).order_by('-created_at')

        total_analyses = analyses.count()

        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses
        else:
            avg_confidence = 0

        from django.core.paginator import Paginator
        paginator = Paginator(analyses, 10)
        page_number = request.GET.get('page')
        analyses_page = paginator.get_page(page_number)

        context = {
            'analyses': analyses_page,
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence * 100, 2),
            'user_role': user_role,
        }
        return render(request, "superuser/admin_test_history.html", context)


class AdminPatientProfileView(RoleRequiredMixin, View):
    """GET /admin/patient/<patient_id> - View patient details and test results"""
    allowed_roles = ('admin',)
    def get(self, request, patient_id):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/admin/patient/' + str(patient_id))

        try:
            patient = Patient.objects.select_related('created_by').get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return redirect('/admin/patients/')

        analyses = Analysis.objects.filter(patient=patient).filter(
            Q(batch_id__isnull=True) | Q(is_batch_primary=True)
        ).order_by('-created_at')
        total_analyses = analyses.count()

        if total_analyses > 0:
            avg_confidence = sum(a.confidence for a in analyses) / total_analyses
            latest_analysis = analyses.first()
        else:
            avg_confidence = 0
            latest_analysis = None

        severity_dist = {}
        for analysis in analyses:
            label = analysis.predicted_label
            severity_dist[label] = severity_dist.get(label, 0) + 1

        severity_distribution = [
            (label, count, round((count / total_analyses) * 100, 1) if total_analyses > 0 else 0)
            for label, count in severity_dist.items()
        ]

        from django.core.paginator import Paginator

        def enrich(qs):
            result = []
            for a in qs:
                result.append({
                    'analysis_id': a.analysis_id,
                    'predicted_class': a.predicted_class,
                    'predicted_label': a.predicted_label,
                    'confidence': a.confidence,
                    'confidence_pct': round(a.confidence * 100, 1),
                    'confidence_level': a.confidence_level,
                    'recommendation': a.recommendation,
                    'image': a.image,
                    'created_at': a.created_at,
                })
            return result

        paginator = Paginator(analyses, 10)
        analyses_page = paginator.get_page(request.GET.get('page'))
        analyses_list = enrich(analyses_page)

        context = {
            'patient': patient,
            'analyses': analyses_list,
            'analyses_page': analyses_page,
            'total_analyses': total_analyses,
            'avg_confidence': round(avg_confidence * 100, 2),
            'latest_analysis': latest_analysis,
            'severity_distribution': severity_distribution,
        }
        return render(request, "superuser/admin_patient_profile.html", context)


from django.contrib.auth.decorators import login_required


@login_required
def admin_assign_doctor(request):
    if request.user.profile.role != 'admin':
        messages.error(request, "Access denied.")
        return redirect('home')
        
    patients = Patient.objects.all()
    doctors = User.objects.filter(profile__role='user')

    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        doctor_id = request.POST.get('doctor_id')
        patient = get_object_or_404(Patient, patient_id=patient_id)
        doctor = get_object_or_404(User, id=doctor_id)
        patient.assigned_doctor = doctor
        patient.save()
        messages.success(request, f"Assigned Dr. {doctor.get_full_name()} to {patient.full_name}")
        return redirect('admin-assign-doctor')

    # Paginate the assignments table (10 rows); the dropdown still uses the full list
    paginator = Paginator(patients.order_by('-created_at'), 10)
    patients_page = paginator.get_page(request.GET.get('page'))

    return render(request, 'superuser/admin_assign_doctor.html', {
        'patients': patients,
        'patients_page': patients_page,
        'doctors': doctors
    })


# ─── Workflow API ───
from rest_framework import permissions


class AssignDoctorView(APIView):
    """Admin assigns a doctor to a patient"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, patient_id):
        if request.user.profile.role != 'admin':
            return Response({"error": "Only admins can assign doctors"}, status=status.HTTP_403_FORBIDDEN)
        
        patient = get_object_or_404(Patient, patient_id=patient_id)
        doctor_id = request.data.get('doctor_id')
        
        if not doctor_id:
            return Response({"error": "doctor_id is required"}, status=status.HTTP_400_BAD_REQUEST)
            
        doctor = get_object_or_404(User, id=doctor_id, profile__role='user')
        patient.assigned_doctor = doctor
        patient.save()
        
        return Response({"message": f"Patient assigned to doctor {doctor.username}"}, status=status.HTTP_200_OK)
