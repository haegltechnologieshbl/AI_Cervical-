"""Role-specific views — shared imports/helpers come from core.views."""
from .views import *  # noqa: F401,F403
from django.contrib.auth.decorators import login_required


@login_required
def technician_dashboard(request):
    if request.user.profile.role != 'senior_technician':
        messages.error(request, "Access denied.")
        return redirect('home')
        
    from django.utils import timezone
    from datetime import datetime, time
    
    assigned_patients = Patient.objects.filter(assigned_technician=request.user)
    total_patients = assigned_patients.count()
    
    pap_smears_completed = PapSmearTest.objects.filter(technician=request.user).count()
    total_analyses = Analysis.objects.filter(created_by=request.user).count()
    
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_submissions = PapSmearTest.objects.filter(
        technician=request.user,
        test_date__gte=today_start
    ).count()
    
    recent_pap_smears = PapSmearTest.objects.filter(
        technician=request.user
    ).select_related('patient').order_by('-test_date')[:5]
    
    context = {
        'patients': assigned_patients,
        'total_patients': total_patients,
        'pap_smears_completed': pap_smears_completed,
        'total_analyses': total_analyses,
        'today_submissions': today_submissions,
        'recent_pap_smears': recent_pap_smears,
    }
    
    return render(request, 'technician/technician_dashboard_new.html', context)


@login_required
def submit_pap_smear(request, patient_id):
    if request.user.profile.role != 'senior_technician':
        messages.error(request, "Access denied.")
        return redirect('home')
        
    patient = get_object_or_404(Patient, patient_id=patient_id, assigned_technician=request.user)

    existing_pap_smear = patient.pap_smear_tests.order_by('-test_date').first()
    latest_analysis = Analysis.objects.filter(patient=patient).order_by('-created_at').first()
    edit_mode = request.GET.get('edit') == '1'

    if request.method == 'POST':
        result = request.POST.get('result')
        notes = request.POST.get('notes', '')
        if existing_pap_smear:
            # Update the existing result (edit).
            existing_pap_smear.result = result
            existing_pap_smear.notes = notes
            existing_pap_smear.save()
            messages.success(request, f"Updated Pap Smear result for {patient.full_name}")
            return redirect('technician-dashboard')
        # New record — AI analysis must be done first.
        if not latest_analysis:
            messages.error(request, "Run the AI analysis before recording the Pap smear result.")
            return redirect(f'/technician/analyze/?patient_id={patient_id}')
        PapSmearTest.objects.create(
            patient=patient,
            technician=request.user,
            result=result,
            notes=notes
        )
        Notification.notify(
            patient.assigned_doctor,
            f"{request.user.get_full_name() or request.user.username} recorded a Pap smear result for {patient.full_name}",
            url=f"/doctor/patient/{patient.patient_id}/review/",
        )
        messages.success(request, f"Recorded Pap Smear result for {patient.full_name}")
        return redirect('technician-dashboard')

    return render(request, 'technician/submit_pap_smear.html', {
        'patient': patient,
        'existing_pap_smear': existing_pap_smear,
        'latest_analysis': latest_analysis,
        'edit_mode': edit_mode,
    })


@login_required
def delete_pap_smear(request, patient_id):
    if request.user.profile.role != 'senior_technician':
        messages.error(request, "Access denied.")
        return redirect('home')
    patient = get_object_or_404(Patient, patient_id=patient_id, assigned_technician=request.user)
    pap = patient.pap_smear_tests.order_by('-test_date').first()
    if pap:
        pap.delete()
        messages.success(request, f"Deleted Pap Smear result for {patient.full_name}")
    else:
        messages.warning(request, "No Pap Smear result to delete.")
    return redirect('technician-dashboard')


# ─── Workflow API ───
from rest_framework import generics, permissions, serializers
from .serializers import PatientSerializer, PapSmearTestSerializer


class TechnicianPatientsView(generics.ListAPIView):
    """technician/technicians fetch their assigned patients"""
    serializer_class = PatientSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if self.request.user.profile.role == 'senior_technician':
            return Patient.objects.filter(assigned_technician=self.request.user)
        return Patient.objects.none()


class PapSmearTestCreateView(generics.CreateAPIView):
    """technician/technicians submit Pap Smear results"""
    serializer_class = PapSmearTestSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        if self.request.user.profile.role != 'senior_technician':
            raise serializers.ValidationError("Only technicians can submit Pap smear results")
            
        patient_id = self.request.data.get('patient')
        patient = get_object_or_404(Patient, patient_id=patient_id)
        
        if patient.assigned_technician != self.request.user:
            raise serializers.ValidationError("You are not assigned to this patient")

        serializer.save(technician=self.request.user)
        Notification.notify(
            patient.assigned_doctor,
            f"{self.request.user.get_full_name() or self.request.user.username} submitted a Pap smear result for {patient.full_name}",
            url=f"/doctor/patient/{patient.patient_id}/review/",
        )
