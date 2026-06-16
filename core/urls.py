from django.urls import path
from . import views

urlpatterns = [
    # --- UI pages (Django templates) ---
    path("", views.HomePageView.as_view(), name="home"),
    path("analyze/", views.AnalyzePageView.as_view(), name="analyze-page"),
    path("api/demo/<str:category>/", views.DemoDatasetView.as_view(), name="demo-dataset"),
    path("history/", views.HistoryPageView.as_view(), name="history-page"),
    path("login/", views.LoginPageView.as_view(), name="login-page"),
    path("dashboard/", views.DashboardPageView.as_view(), name="dashboard-page"),
    path("profile/", views.UserProfilePageView.as_view(), name="user-profile-page"),
    path("patient/dashboard/", views.PatientDashboardView.as_view(), name="patient-dashboard"),
    path("analysis/<uuid:analysis_id>", views.AnalysisDetailView.as_view(), name="analysis-detail-page"),
    path("settings/", views.SettingsPageView.as_view(), name="settings-page"),

    # --- Admin pages ---
    path("admin/", views.AdminDashboardView.as_view(), name="admin-dashboard"),
    path("admin/analysis/<uuid:analysis_id>", views.AdminAnalysisDetailView.as_view(), name="admin-analysis-detail"),

    path("admin/doctors/", views.AdminDoctorsView.as_view(), name="admin-doctors"),
    # urls.py
    path("admin/doctor/add/", views.AddDoctorView.as_view(), name="add-doctor"),
    path("admin/user/<int:user_id>/edit/", views.EditDoctorView.as_view(), name="admin-edit-doctor"),


    path("admin/doctors/<int:user_id>/", views.AdminDoctorsDetailView.as_view(), name="admin-doctors-detail"),
    
    path("admin/doctors/<int:user_id>/history/", views.AdminUserHistoryView.as_view(), name="admin-user-history"),

    # --- Senior Technician CRUD ---
    path("admin/technicians/", views.AdminTechniciansView.as_view(), name="admin-technicians"),
    path("admin/technician/add/", views.AddTechnicianView.as_view(), name="add-technician"),
    path("admin/technician/<int:user_id>/edit/", views.EditTechnicianView.as_view(), name="edit-technician"),
    path("admin/patients/", views.AdminPatientsView.as_view(), name="admin-patients"),
    path("admin/test-results/", views.AdminTestResultsView.as_view(), name="admin-test-results"),
    path("admin/test-history/", views.AdminTestHistoryView.as_view(), name="admin-test-history"),
    path("admin/patient/<uuid:patient_id>", views.AdminPatientProfileView.as_view(), name="admin-patient-profile"),
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
    path("api/v1/dashboard/hospital-stats", views.HospitalDashboardStatsView.as_view(), name="hospital-dashboard-stats"),

    # --- Patient Management API ---
    path("api/v1/patients", views.PatientListCreateAPIView.as_view(), name="patient-list"),
    path("api/v1/patient/<uuid:patient_id>/create-user", views.CreatePatientUserView.as_view(), name="create-patient-user"),
    path("api/v1/patients/<uuid:patient_id>", views.PatientDetailAPIView.as_view(), name="patient-detail"),
    path("api/v1/patients/<uuid:patient_id>/history", views.PatientHistoryAPIView.as_view(), name="patient-history"),
    path("api/v1/patients/<uuid:patient_id>/stats", views.PatientStatsAPIView.as_view(), name="patient-stats"),

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
]
