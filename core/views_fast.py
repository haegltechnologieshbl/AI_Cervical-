"""
Add this to your views.py to enable fast analysis endpoint.

Add this import near the top of views.py:
    from services.inference_fast import FastInferenceService

Add this new view class after AnalyzeView:

@method_decorator(csrf_exempt, name='dispatch')
class FastAnalyzeView(APIView):
    \"\"\"POST /api/v1/analyze/fast - Fast analysis for large images.
    Uses optimized preprocessing for faster results.\"\"\"
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Get all files from request
        files = request.FILES.getlist('file')
        MAX_FILES = 200

        if len(files) > MAX_FILES:
            return Response(
                {"error": f"Maximum {MAX_FILES} images allowed"},
                status=400
            )
        if not files:
            return Response(
                {"error": "No image files provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if heatmap generation should be skipped (default: yes for speed)
        skip_heatmap = request.query_params.get('skip_heatmap', 'true').lower() in ('true', '1', 'yes')

        print(f"[FAST] Received {len(files)} files for fast analysis")

        allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
        svc = FastInferenceService.get()

        if svc.session is None:
            return Response(
                {"error": "Model not loaded. Please contact support."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        predictions_list = []
        analysis_records = []

        # Process each uploaded image
        for idx, image_file in enumerate(files):
            # Validate MIME type
            if image_file.content_type not in allowed_types:
                return Response(
                    {"error": f"File {idx+1} ({image_file.name}): Unsupported type {image_file.content_type}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                print(f"[FAST] Processing file {idx+1}/{len(files)}: {image_file.name}")
                prediction = svc.predict(
                    image_file,
                    skip_heatmap=skip_heatmap
                )
                predictions_list.append(prediction)

                # Store individual analysis record in database
                image_file.seek(0)
                extra_fields = {}
                for key in ("stage_label", "risk_level", "risk_color", "explanation", "confidence_interpretation"):
                    extra_fields[key] = prediction.pop(key)

                image_file.seek(0)
                analysis = Analysis.objects.create(
                    created_by=request.user,
                    image=image_file,
                    **prediction,
                )
                analysis_records.append(analysis)
                print(f"[FAST] ✓ Stored analysis for file {idx+1}")

            except RuntimeError as e:
                return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Average predictions across all images
        print(f"[FAST] Averaging {len(predictions_list)} predictions")
        averaged_prediction = svc.average_predictions(predictions_list)

        # Build response with individual results
        individual_results = []
        for record, pred in zip(analysis_records, predictions_list):
            individual_results.append({
                "analysis_id": str(record.analysis_id),
                "predicted_class": pred["predicted_class"],
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"],
                "image_url": record.image.url if record.image else None,
            })

        resp = {
            "predicted_class": averaged_prediction["predicted_class"],
            "predicted_label": averaged_prediction["predicted_label"],
            "stage_label": averaged_prediction["stage_label"],
            "probabilities": averaged_prediction["probabilities"],
            "confidence": averaged_prediction["confidence"],
            "uncertainty": averaged_prediction["uncertainty"],
            "confidence_level": averaged_prediction["confidence_level"],
            "risk_level": averaged_prediction["risk_level"],
            "risk_color": averaged_prediction["risk_color"],
            "confidence_interpretation": averaged_prediction["confidence_interpretation"],
            "recommendation": averaged_prediction["recommendation"],
            "explanation": averaged_prediction["explanation"],
            "images_analyzed": len(files),
            "analysis_id": str(analysis_records[0].analysis_id) if analysis_records else None,
            "individual_analyses": [str(a.analysis_id) for a in analysis_records],
            "individual_results": individual_results,
        }

        return Response(resp, status=status.HTTP_201_CREATED)

"""

# Add this to urls.py:
# path("api/v1/analyze/fast", views.FastAnalyzeView.as_view(), name="analyze-fast"),
