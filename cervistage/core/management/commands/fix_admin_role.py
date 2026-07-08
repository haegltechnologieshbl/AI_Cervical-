from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model


class Command(BaseCommand):
    help = 'Fix admin user role from user to admin'

    def handle(self, *args, **options):
        User = get_user_model()

        # Find admin user by email
        admin_email = 'admin@gmail.com'
        try:
            admin = User.objects.get(email=admin_email)
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User with email {admin_email} not found')
            )
            return

        # Show current state
        current_role = admin.profile.role if hasattr(admin, 'profile') else 'No profile'
        self.stdout.write(f'Current role for {admin.username} ({admin_email}): {current_role}')

        # Update role to admin
        admin.profile.role = 'admin'
        admin.profile.save()

        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated role to: {admin.profile.role}')
        )

        # Show all users and their roles for verification
        self.stdout.write('\n=== All Users and Their Roles ===')
        all_users = User.objects.all()
        for user in all_users:
            role = user.profile.role if hasattr(user, 'profile') else 'No profile'
            self.stdout.write(f'  - {user.username} ({user.email}): {role}')
