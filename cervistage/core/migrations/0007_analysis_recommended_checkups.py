# Generated manually for recommended patient checkups

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_patient_user_account_alter_userprofile_role'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysis',
            name='recommended_checkups',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
