# Template Structure - CerviStage AI (Clean Version)

## 🎯 Only 2 Dashboards

### 1. `user_dashboard.html` 
**Route:** `/dashboard/`
**View:** `DashboardPageView`
**Access:** Clinicians only
**Base:** `user_base.html` (navbar)
**Purpose:** Personal dashboard for clinicians
- My Patients count
- Total Analyses count
- Recent activity (last 7 days)
- Quick actions (New Analysis, Add Patient, View History)
- Recent activity table

### 2. `admin_dashboard.html`
**Route:** `/admin/`
**View:** `AdminDashboardView`
**Access:** Admins only
**Base:** `admin_base.html` (sidebar)
**Purpose:** System-wide dashboard for admins
- Total users, patients, analyses
- User role distribution
- Recent system activity
- Recent user registrations

---

## Base Templates (3)

### 1. `base.html`
**Purpose:** General-purpose base with top navbar
**Used by:** Public pages, home, analyze, patients, etc.
**Navigation:** Top navbar
- **Features:** Logo, navigation, user menu, footer

### 2. `user_base.html`
**Purpose:** Base template for clinicians/users with navbar
**Used by:** Login, user dashboard
**Navigation:** Top navbar
- **Links:** Analyze, Dashboard, History, Settings
- **User Menu:** My Profile, Settings, Logout

### 3. `admin_base.html`
**Purpose:** Base template for admin users with sidebar
**Used by:** All admin pages
**Navigation:** Left sidebar
- **Links:** Dashboard, Users List, All Patients, Test Results, Test History, Settings

---

## 📊 Dashboard Flow

### Clinicians (Regular Users)
```
Login → /dashboard/ (User Dashboard with navbar)
├── Analyze (/)
├── History (/history/)
├── My Profile (/profile/)
└── Settings (/settings/)
```

### Admins
```
Login → /admin/ (Admin Dashboard with sidebar)
├── Users List (/admin/users/)
├── All Patients (/admin/patients/)
├── User Test Results
├── User Test History
└── Settings (/settings/)
```

---

## ✅ File Cleanup Complete

### Removed:
- ❌ `dashboard.html` (old normal dashboard)

### Kept:
- ✅ `user_dashboard.html` (for clinicians at /dashboard/)
- ✅ `admin_dashboard.html` (for admins at /admin/)

---

## 📁 Template Count: 19 Files

### User Pages (using base.html or user_base.html)
1. user_base.html
2. base.html
3. home.html
4. analyze.html
5. user_dashboard.html ⭐
6. history.html
7. patients.html
8. patient_profile.html
9. analysis_detail.html
10. user_profile.html
11. settings.html
12. login.html
13. register.html

### Admin Pages (using admin_base.html)
14. admin_base.html
15. admin_dashboard.html ⭐
16. admin_users.html
17. admin_user_history.html
18. admin_patients.html
19. admin_patient_profile.html
20. admin_analysis_detail.html

---

## 🎉 Final Result

**Clean Dashboard Structure:**
- 1 User Dashboard (with navbar) for clinicians
- 1 Admin Dashboard (with sidebar) for admins
- No redundant or extra dashboards
- Clear separation of concerns
