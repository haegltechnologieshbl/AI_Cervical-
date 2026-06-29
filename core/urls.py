from django.urls import path
from . import views
from . import views_superuser
from . import views_patient
from . import views_doctor
from . import views_technician

# Feedback imports
from .views_superuser import AdminFeedbackView, FeedbackDetailView, FeedbackExportView
from .views import FeedbackSubmitView

urlpatterns = [
    # --- UI pages (Django templates) ---
    path("", views.HomePageView.as_view(), name="home"),
    path("analyze/", views.AnalyzePageView.as_view(), name="analyze-page"),
    path("technician/analyze/", views.AnalyzePageView.as_view(), name="technician-analyze-page"),
    path("api/demo/<str:category>/", views.DemoDatasetView.as_view(), name="demo-dataset"),
    path("history/", views.HistoryPageView.as_view(), name="history-page"),
    path("login/", views.LoginPageView.as_view(), name="login-page"),
    path("dashboard/", views.DashboardPageView.as_view(), name="dashboard-page"),
    path("profile/", views.UserProfilePageView.as_view(), name="user-profile-page"),
    path("patient/dashboard/", views_patient.PatientDashboardView.as_view(), name="patient-dashboard"),
    path("doctor/dashboard/", views_doctor.DoctorDashboardView.as_view(), name="doctor-dashboard"),

    # --- Doctor pages (role-specific URLs) ---
    path("doctor/patients/", views.PatientManagementView.as_view(), name="doctor-patients"),
    path("doctor/patient/<uuid:patient_id>", views.PatientProfileView.as_view(), name="doctor-patient-profile"),
    path("doctor/history/", views.HistoryPageView.as_view(), name="doctor-history"),
    path("doctor/analysis/<uuid:analysis_id>", views.AnalysisDetailView.as_view(), name="doctor-analysis-detail"),
    path("doctor/patient/<uuid:patient_id>/notes/add/", views.AddClinicalNoteView.as_view(), name="doctor-add-clinical-note"),

    # --- Technician pages (role-specific URLs) ---
    path("technician/patients/", views.PatientManagementView.as_view(), name="technician-patients-page"),
    path("technician/patient/<uuid:patient_id>", views.PatientProfileView.as_view(), name="technician-patient-profile"),
    path("technician/history/", views.HistoryPageView.as_view(), name="technician-history"),
    path("technician/analysis/<uuid:analysis_id>", views.AnalysisDetailView.as_view(), name="technician-analysis-detail"),
    path("technician/patient/<uuid:patient_id>/notes/add/", views.AddClinicalNoteView.as_view(), name="technician-add-clinical-note"),
    path("analysis/<uuid:analysis_id>", views.AnalysisDetailView.as_view(), name="analysis-detail-page"),
    path("analysis/<uuid:analysis_id>/delete/", views.DeleteAnalysisView.as_view(), name="analysis-delete"),

    # --- Workflow UI pages ---
    path("admin/assign-doctor/", views_superuser.admin_assign_doctor, name="admin-assign-doctor"),
    path("doctor/assign-technician/", views_doctor.doctor_assign_technician, name="doctor-assign-technician"),
    path("technician/dashboard/", views_technician.technician_dashboard, name="technician-dashboard"),
    path("technician/patient/<uuid:patient_id>/pap-smear/", views_technician.submit_pap_smear, name="submit-pap-smear"),
    path("technician/patient/<uuid:patient_id>/pap-smear/delete/", views_technician.delete_pap_smear, name="delete-pap-smear"),
    path("doctor/patient/<uuid:patient_id>/review/", views_doctor.doctor_review_checkups, name="doctor-review-checkups"),

    # --- Admin pages ---
    path("admin/", views_superuser.AdminDashboardView.as_view(), name="admin-dashboard"),
    path("admin/analysis/<uuid:analysis_id>", views_superuser.AdminAnalysisDetailView.as_view(), name="admin-analysis-detail"),

    path("admin/doctors/", views_superuser.AdminDoctorsView.as_view(), name="admin-doctors"),
    # urls.py
    path("admin/doctor/add/", views_superuser.AddDoctorView.as_view(), name="add-doctor"),
    path("admin/user/<int:user_id>/edit/", views_superuser.EditDoctorView.as_view(), name="admin-edit-doctor"),


    path("admin/doctors/<int:user_id>/", views_superuser.AdminDoctorsDetailView.as_view(), name="admin-doctors-detail"),
    
    path("admin/doctors/<int:user_id>/history/", views_superuser.AdminUserHistoryView.as_view(), name="admin-user-history"),

    # --- Senior Technician CRUD ---
    path("admin/technicians/", views_superuser.AdminTechniciansView.as_view(), name="admin-technicians"),
    path("admin/technician/add/", views_superuser.AddTechnicianView.as_view(), name="add-technician"),
    path("admin/technician/<int:user_id>/edit/", views_superuser.EditTechnicianView.as_view(), name="edit-technician"),
    path("admin/patients/", views_superuser.AdminPatientsView.as_view(), name="admin-patients"),
    path("admin/test-results/", views_superuser.AdminTestResultsView.as_view(), name="admin-test-results"),
    path("admin/test-history/", views_superuser.AdminTestHistoryView.as_view(), name="admin-test-history"),
    path("admin/patient/<uuid:patient_id>", views_superuser.AdminPatientProfileView.as_view(), name="admin-patient-profile"),
    path("patients/", views.PatientManagementView.as_view(), name="patient-management"),
    path("patient/<uuid:patient_id>", views.PatientProfileView.as_view(), name="patient-profile"),
    path("patient/add/", views.AddPatientView.as_view(), name="add-patient"),
    path("patient/<uuid:patient_id>/notes/add/", views.AddClinicalNoteView.as_view(), name="add-clinical-note"),

    path("admin/patient/edit/<uuid:patient_id>/", views.AddPatientView.as_view(), name="edit-patient"),


    # --- REST API ---
    path("api/v1/analyze", views.AnalyzeView.as_view(), name="analyze"),
    path("api/v1/analyze/fast", views.FastAnalyzeView.as_view(), name="analyze-fast"),
    path("api/v1/analyze/vps", views.VPSAnalyzeView.as_view(), name="analyze-vps"),
    path("api/v1/analyze/<uuid:analysis_id>", views.AnalysisApiDetailView.as_view(), name="analyze-detail"),
    path("api/v1/reports/generate", views.ReportView.as_view(), name="report-generate"),
    path("api/v1/health", views.HealthView.as_view(), name="health"),
    path("api/v1/stats", views.StatsView.as_view(), name="stats"),

    # --- Notifications ---
    path("api/v1/notifications", views.NotificationListView.as_view(), name="notifications"),
    path("api/v1/notifications/mark-read", views.NotificationMarkReadView.as_view(), name="notifications-mark-read"),
    path("api/v1/dashboard/hospital-stats", views.HospitalDashboardStatsView.as_view(), name="hospital-dashboard-stats"),

    # --- Patient Management API ---
    path("api/v1/patients", views.PatientListCreateAPIView.as_view(), name="patient-list"),
    path("api/v1/patient/<uuid:patient_id>/create-user", views.CreatePatientUserView.as_view(), name="create-patient-user"),
    path("api/v1/patients/<uuid:patient_id>", views.PatientDetailAPIView.as_view(), name="patient-detail"),
    path("api/v1/patients/<uuid:patient_id>/history", views.PatientHistoryAPIView.as_view(), name="patient-history"),
    path("api/v1/patients/<uuid:patient_id>/stats", views.PatientStatsAPIView.as_view(), name="patient-stats"),
    
    # --- Workflow API ---
    path("api/v1/workflow/patient/<uuid:patient_id>/assign-doctor", views_superuser.AssignDoctorView.as_view(), name="assign-doctor-api"),
    path("api/v1/workflow/patient/<uuid:patient_id>/assign-technician", views_doctor.AssignTechnicianView.as_view(), name="assign-technician-api"),
    path("api/v1/workflow/technician/patients", views_technician.TechnicianPatientsView.as_view(), name="technician-patients-api"),
    path("api/v1/workflow/pap-smear", views_technician.PapSmearTestCreateView.as_view(), name="pap-smear-create-api"),
    path("api/v1/workflow/checkups", views_doctor.DoctorCheckupCreateView.as_view(), name="doctor-checkups-create-api"),

    # --- Authentication API ---
    path("api/v1/auth/login", views.LoginView.as_view(), name="login"),
    path("api/v1/auth/logout", views.LogoutView.as_view(), name="logout"),
    path("api/v1/token/refresh/", views.TokenRefreshView.as_view(), name="token-refresh"),
    path("api/v1/auth/profile", views.UserProfileView.as_view(), name="user-profile"),
    path("api/v1/auth/change-password", views.ChangePasswordView.as_view(), name="change-password"),
    path("api/v1/auth/user/<int:user_id>", views.UserManagementView.as_view(), name="user-management"),
       
    # --- Password Reset ---
    path("forgot-password/", views.ForgotPasswordView.as_view(), name="forgot-password"),
    path("verify-otp/", views.VerifyOTPView.as_view(), name="verify-otp"),

    # --- Feedback ---
    path('feedback/submit/', FeedbackSubmitView.as_view(), name='feedback-submit'),
    path('admin/feedback/', AdminFeedbackView.as_view(), name='admin-feedback'),
    path('admin/feedback/<uuid:feedback_id>/', FeedbackDetailView.as_view(), name='feedback-detail'),
    path('admin/feedback/export/', FeedbackExportView.as_view(), name='feedback-export'),
]
