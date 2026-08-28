# Pose model

The application uses Google's MediaPipe Pose Landmarker **Full** model bundle
(`pose_landmarker_full.task`) for its 33-landmark live overlay. The Lite bundle is retained as a
fallback for lower-powered machines. On the development laptop, a local camera benchmark showed
no material throughput penalty for Full (about 31 FPS for both variants), so Full was selected for
the clearer joint estimate.

Model-derived coordinates and NeuroFlex scores remain investigational until benchmarked against
the project validation datasets.
