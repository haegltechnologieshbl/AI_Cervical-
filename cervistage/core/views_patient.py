"""Role-specific views — shared imports/helpers come from core.views."""
from .views import *  # noqa: F401,F403


class PatientDashboardView(RoleRequiredMixin, View):
    """GET /patient/dashboard/ - Patient-specific dashboard showing only their results"""
    allowed_roles = ('patient',)
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/login/?next=/patient/dashboard/')

        # Only allow patients to access this dashboard
        if request.user.profile.role != 'patient':
            return redirect('/dashboard/')

        # Get patient record linked to this user
        try:
            patient = request.user.patient_record
        except Patient.DoesNotExist:
            return redirect('/login/')

        # Get only this patient's analyses
        analyses_qs = Analysis.objects.filter(
            patient=patient
        ).filter(
            Q(batch_id__isnull=True) | Q(is_batch_primary=True)
        ).select_related('created_by', 'created_by__profile').order_by('-created_at')

        total_analyses = analyses_qs.count()
        latest_analysis = analyses_qs.first() if analyses_qs else None
        if latest_analysis:
            latest_analysis.confidence_pct = round(latest_analysis.confidence * 100)

        from django.core.paginator import Paginator
        import json as _json
        paginator = Paginator(analyses_qs, 10)
        page_number = request.GET.get('page', 1)
        analyses_page = paginator.get_page(page_number)

        # Enrich page items
        for a in analyses_page:
            a.confidence_pct = round(a.confidence * 100)

        # Build chart data from ALL analyses (not just current page)
        all_analyses = list(analyses_qs.values('predicted_class', 'predicted_label', 'confidence', 'created_at'))

        # Stage distribution counts
        stage_counts = {}
        for a in all_analyses:
            lbl = a['predicted_label']
            stage_counts[lbl] = stage_counts.get(lbl, 0) + 1

        stage_labels = list(stage_counts.keys())
        stage_data   = [stage_counts[k] for k in stage_labels]
        stage_colors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444']
        stage_color_map = {
            'Normal': '#22c55e', 'Atypia': '#84cc16',
            'LSIL': '#eab308', 'HSIL': '#f97316', 'Carcinoma': '#ef4444',
        }
        stage_bg = [stage_color_map.get(lbl, '#8b5cf6') for lbl in stage_labels]

        # Confidence trend (last 10 for chart, oldest first)
        trend_items = list(reversed(all_analyses[:10]))
        trend_labels = [a['created_at'].strftime('%d %b') for a in trend_items]
        trend_data   = [round(a['confidence'] * 100, 1) for a in trend_items]

        # Recent checkups: the tests doctors recommended after reviewing this patient
        recent_checkups = []
        latest_analysis_id = str(latest_analysis.analysis_id) if latest_analysis else ''
        for rec in patient.checkup_recommendations.order_by('-created_at')[:5]:
            for name in rec.recommended_list():
                recent_checkups.append({
                    'name': name,
                    'date': rec.created_at.strftime('%d %b %Y'),
                    'analysis_id': latest_analysis_id,
                })
            if len(recent_checkups) >= 8:
                break

        # Latest doctor checkup recommendation (distinct test names) for this patient
        latest_checkup = patient.checkup_recommendations.order_by('-created_at').first()
        doctor_checkups = latest_checkup.recommended_list() if latest_checkup else []

        context = {
            'patient': patient,
            'analyses': analyses_page,
            'total_analyses': total_analyses,
            'latest_analysis': latest_analysis,
            # chart data (JSON safe)
            'chart_stage_labels': _json.dumps(stage_labels),
            'chart_stage_data':   _json.dumps(stage_data),
            'chart_stage_bg':     _json.dumps(stage_bg),
            'chart_trend_labels': _json.dumps(trend_labels),
            'chart_trend_data':   _json.dumps(trend_data),
            'recent_checkups':    recent_checkups,
            'doctor_checkups':    doctor_checkups,
        }
        return render(request, "patient/patient_dashboard.html", context)
