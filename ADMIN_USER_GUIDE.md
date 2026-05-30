# Admin User Creation Guide

## User Roles

CerviStage AI has two types of users:

### 1. Regular Users (Clinician)
- Created through the web registration form at `/register/`
- Default role: `clinician`
- Can access: Analyze, Patients, History, Dashboard, Settings
- **Cannot** access Django admin panel

### 2. Admin Users
- Created **only** via Django's `createsuperuser` command
- Has access to all features including Django admin panel
- Can manage users, patients, analyses, and system settings

## Creating an Admin User

### Method 1: Using Django Command (Recommended)

Open your terminal in the backend directory and run:

```bash
cd backend
python manage.py createsuperuser
```

You'll be prompted to enter:
- Username
- Email address
- Password (twice for confirmation)

**Example:**
```
Username: admin
Email address: admin@cervistage.ai
Password: ********
Password (again): ********
Superuser created successfully.
```

### Method 2: Using Python Shell

```bash
cd backend
python manage.py shell
```

Then run:
```python
from django.contrib.auth import get_user_model

User = get_user_model()

# Create superuser
admin = User.objects.create_superuser(
    username='admin',
    email='admin@cervistage.ai',
    password='your_secure_password',
    first_name='System',
    last_name='Administrator'
)

# Set role
admin.profile.role = 'admin'
admin.profile.save()

print("Admin user created successfully!")
exit()
```

## Accessing Admin Panel

Once you've created a superuser:

1. Go to: `http://127.0.0.1:8000/admin/`
2. Login with your superuser credentials
3. You'll have full access to:
   - User management
   - Patient records
   - Analysis history
   - System configuration

## User Role Permissions

| Feature | Clinician | Admin |
|---------|-----------|-------|
| Analyze images | ✅ | ✅ |
| View dashboard | ✅ | ✅ |
| Manage patients | ✅ | ✅ |
| View history | ✅ | ✅ |
| Access settings | ✅ | ✅ |
| Django admin | ❌ | ✅ |
| Manage users | ❌ | ✅ |

## Security Best Practices

1. **Use strong passwords** for admin accounts (minimum 12 characters, mix of letters, numbers, symbols)
2. **Limit admin accounts** - only create as many as needed
3. **Change default passwords** regularly
4. **Never share admin credentials** via email or chat
5. **Use environment variables** for production deployments

## Troubleshooting

### "Command not found: manage.py"
Make sure you're in the backend directory:
```bash
cd backend
python manage.py createsuperuser
```

### "User already exists"
If you need to reset an admin password:
```bash
python manage.py changepassword <username>
```

### Can't access admin panel
1. Verify the user is a superuser:
```bash
python manage.py shell
```
```python
from django.contrib.auth import get_user_model
User = get_user_model()
user = User.objects.get(username='admin')
print(user.is_superuser)  # Should print: True
```

2. If `False`, promote the user:
```python
user.is_superuser = True
user.is_staff = True
user.save()
```

---

**Last Updated:** April 7, 2026
