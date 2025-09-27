# DriveSense
DriveSense

## MediaPipe Face Landmarker model

The drowsiness analyzer now combines ML Kit probabilities with MediaPipe eye aspect ratios.
The MediaPipe Tasks runtime expects the `face_landmarker.task` model to live in
`app/src/main/assets/`. The binary is **not** committed to the repository to keep
the history lightweight and avoid large binaries in diffs.

During Gradle builds the `downloadFaceLandmarkerModel` task fetches the latest
MediaPipe model automatically. You can also trigger it manually:

```bash
./gradlew :app:downloadFaceLandmarkerModel
```

If you prefer to manage the file yourself, place the model at the path above.
