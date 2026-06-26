"""Role-specific views — shared imports/helpers come from core.views."""
from .views import *  # noqa: F401,F403
from django.contrib.auth.decorators import login_required


class DoctorDashboardView(RoleRequiredMixin, View):
    """GET /doctor/dashboard/ - Doctor dashboard with analytics and pending reviews."""
    allowed_roles = ('user',)  # 'user' is the Doctor role

    def get(self, request):
        return render(request, "doctor/doctor_dashboard.html", build_doctor_dashboard_context(request.user))


@login_required
def doctor_assign_technician(request):
    if request.user.profile.role != 'user':
        messages.error(request, "Access denied.")
        return redirect('home')
        
    patients = Patient.objects.filter(assigned_doctor=request.user)
    technicians = User.objects.filter(profile__role='senior_technician')
    
    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        action = request.POST.get('action', 'assign')
        
        if action == 'unassign':
            patient = get_object_or_404(Patient, patient_id=patient_id, assigned_doctor=request.user)
            patient.assigned_technician = None
            patient.save()
            messages.success(request, f"Unassigned technician from {patient.full_name}")
            return redirect('doctor-assign-technician')
        else:
            technician_id = request.POST.get('technician_id')
            patient = get_object_or_404(Patient, patient_id=patient_id, assigned_doctor=request.user)
            technician = get_object_or_404(User, id=technician_id)
            patient.assigned_technician = technician
            patient.save()
            messages.success(request, f"Assigned technician {technician.get_full_name()} to {patient.full_name}")
            return redirect('doctor-assign-technician')
        
    return render(request, 'doctor/doctor_assign_technician.html', {
        'patients': patients,
        'technicians': technicians
    })


@login_required
def doctor_review_checkups(request, patient_id):
    if request.user.profile.role != 'user':
        messages.error(request, "Access denied.")
        return redirect('home')
        
    patient = get_object_or_404(Patient, patient_id=patient_id, assigned_doctor=request.user)
    pap_smear = patient.pap_smear_tests.order_by('-test_date').first()

    if not pap_smear:
        messages.warning(request, "No Pap Smear test available to review yet.")
        return redirect('doctor-patients')

    # Existing checkup for this Pap Smear (if the doctor already reviewed it)
    existing = pap_smear.associated_checkups.order_by('-created_at').first()

    if request.method == 'POST':
        action = request.POST.get('action', 'save')

        if action == 'delete':
            if existing:
                existing.delete()
                messages.success(request, f"Checkup recommendations removed for {patient.full_name}")
            else:
                messages.warning(request, "No checkup recommendations to remove.")
            return redirect('doctor-review-checkups', patient_id=patient.patient_id)

        fields = dict(
            hpv_pcr_testing=request.POST.get('hpv_pcr') == 'on',
            colposcopy=request.POST.get('colposcopy') == 'on',
            histopathology=request.POST.get('histopathology') == 'on',
            mri_scan=request.POST.get('mri_scan') == 'on',
            treatment_follow_up=request.POST.get('treatment_follow_up', ''),
        )

        if existing:
            for key, value in fields.items():
                setattr(existing, key, value)
            existing.save()
            messages.success(request, f"Updated checkup recommendations for {patient.full_name}")
        else:
            CheckupRecommendation.objects.create(
                patient=patient, doctor=request.user, pap_smear_test=pap_smear, **fields
            )
            messages.success(request, f"Saved checkup recommendations for {patient.full_name}")
        return redirect('doctor-patients')

    return render(request, 'doctor/doctor_review_checkups.html', {
        'patient': patient,
        'pap_smear': pap_smear,
        'checkup': existing,
    })


# ─── Workflow API ───
from rest_framework import generics, permissions, serializers
from .serializers import CheckupRecommendationSerializer


class AssignTechnicianView(APIView):
    """Doctor assigns a technician to a patient"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, patient_id):
        if request.user.profile.role != 'user': # 'user' is the Doctor role
            return Response({"error": "Only doctors can assign technicians"}, status=status.HTTP_403_FORBIDDEN)
            
        patient = get_object_or_404(Patient, patient_id=patient_id)
        
        # Ensure doctor is assigned to this patient
        if patient.assigned_doctor != request.user:
            return Response({"error": "You are not assigned to this patient"}, status=status.HTTP_403_FORBIDDEN)
            
        technician_id = request.data.get('technician_id')
        if not technician_id:
            return Response({"error": "technician/technician_id is required"}, status=status.HTTP_400_BAD_REQUEST)
            
        technician = get_object_or_404(User, id=technician_id, profile__role='senior_technician')
        patient.assigned_technician = technician
        patient.save()
        
        return Response({"message": f"Patient assigned to technician {technician.username}"}, status=status.HTTP_200_OK)


class DoctorCheckupCreateView(generics.CreateAPIView):
    """Doctors review and add checkups"""
    serializer_class = CheckupRecommendationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        if self.request.user.profile.role != 'user': # Doctor
            raise serializers.ValidationError("Only doctors can add checkups")
            
        patient_id = self.request.data.get('patient')
        patient = get_object_or_404(Patient, patient_id=patient_id)
        
        if patient.assigned_doctor != self.request.user:
            raise serializers.ValidationError("You are not assigned to this patient")
            
        # Ensure there is a PapSmearTest done before recommending checkups
        pap_smear = PapSmearTest.objects.filter(patient=patient).order_by('-test_date').first()
        if not pap_smear:
             raise serializers.ValidationError("Cannot recommend checkups without a Pap Smear test result")

        serializer.save(doctor=self.request.user, pap_smear_test=pap_smear)
