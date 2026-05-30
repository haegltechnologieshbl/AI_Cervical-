from django.urls import path
from . import views

urlpatterns = [
    # --- UI pages (Django templates) ---
    path("", views.HomePageView.as_view(), name="home"),
    path("analyze/", views.AnalyzePageView.as_view(), name="analyze-page"),
    path("history/", views.HistoryPageView.as_view(), name="history-page"),
    path("login/", views.LoginPageView.as_view(), name="login-page"),
    path("register/", views.RegisterPageView.as_view(), name="register-page"),
    path("dashboard/", views.DashboardPageView.as_view(), name="dashboard-page"),
    path("profile/", views.UserProfilePageView.as_view(), name="user-profile-page"),
    path("analysis/<uuid:analysis_id>", views.AnalysisDetailView.as_view(), name="analysis-detail-page"),
    path("settings/", views.SettingsPageView.as_view(), name="settings-page"),

    # --- Admin pages ---
    path("admin/", views.AdminDashboardView.as_view(), name="admin-dashboard"),
    path("admin/analysis/<uuid:analysis_id>", views.AdminAnalysisDetailView.as_view(), name="admin-analysis-detail"),
    path("admin/users/", views.AdminUsersView.as_view(), name="admin-users"),
    path("admin/user/<int:user_id>/history/", views.AdminUserHistoryView.as_view(), name="admin-user-history"),
    path("admin/registered-users/", views.AdminRegisteredUsersView.as_view(), name="admin-registered-users"),
    path("admin/test-results/", views.AdminTestResultsView.as_view(), name="admin-test-results"),
    path("admin/test-history/", views.AdminTestHistoryView.as_view(), name="admin-test-history"),

    # --- REST API ---
    path("api/v1/analyze", views.AnalyzeView.as_view(), name="analyze"),
    path("api/v1/analyze/<uuid:analysis_id>", views.AnalysisApiDetailView.as_view(), name="analyze-detail"),
    path("api/v1/reports/generate", views.ReportView.as_view(), name="report-generate"),
    path("api/v1/health", views.HealthView.as_view(), name="health"),
    path("api/v1/stats", views.StatsView.as_view(), name="stats"),

    # --- Authentication API ---
    path("api/v1/auth/register", views.RegisterView.as_view(), name="register"),
    path("api/v1/auth/login", views.LoginView.as_view(), name="login"),
    path("api/v1/auth/logout", views.LogoutView.as_view(), name="logout"),
    path("api/v1/auth/profile", views.UserProfileView.as_view(), name="user-profile"),
    path("api/v1/auth/change-password", views.ChangePasswordView.as_view(), name="change-password"),
    path("api/v1/auth/user/<int:user_id>", views.UserManagementView.as_view(), name="user-management"),
       
]
