package com.drivesense.drivesense

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.Surface
import android.view.View
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import com.drivesense.drivesense.databinding.ActivityMainBinding
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.text.DateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null
    private var faceDetector: FaceDetector? = null
    private var frontAnalyzer: DriveSenseAnalyzer? = null
    private var rearAnalyzer: RoadObjectAnalyzer? = null
    private var toneGenerator: ToneGenerator? = null
    private var lastAlertTimestamp: Long = 0L
    private lateinit var drivingStatusMonitor: DrivingStatusMonitor
    private var drivingDetectionStatus: DrivingDetectionStatus = DrivingDetectionStatus.UNKNOWN
    private var requireDrivingCheck: Boolean = true
    private var latestDriverState: DriverState = DriverState.Initializing
    private var isRoadDetectionEnabled: Boolean = true

    private val frontFpsTracker = FpsTracker()
    private val rearFpsTracker = FpsTracker()

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCameraPipeline()
            } else {
                handleDriverState(DriverState.Error(getString(R.string.status_permission_denied)))
            }
        }

    private val activityRecognitionPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                drivingStatusMonitor.start()
            } else {
                drivingStatusMonitor.handlePermissionDenied()
            }
        }

    private val mainThreadExecutor: Executor by lazy { ContextCompat.getMainExecutor(this) }
    private val settings by lazy { getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.lastEventText.visibility = View.GONE
        binding.frontFpsText.text = getString(R.string.fps_placeholder)
        binding.rearFpsText.text = getString(R.string.fps_placeholder)
        binding.frontPreviewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        binding.rearPreviewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        binding.frontPreviewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        binding.rearPreviewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE

        requireDrivingCheck = settings.getBoolean(KEY_REQUIRE_DRIVING_CHECK, true)
        isRoadDetectionEnabled = settings.getBoolean(KEY_ROAD_DETECTION_ENABLED, true)

        drivingStatusMonitor = DrivingStatusMonitor(
            context = this,
            callbackExecutor = mainThreadExecutor,
            onStatusChanged = ::onDrivingStatusChanged
        )

        binding.drivingCheckSwitch.isChecked = requireDrivingCheck
        binding.drivingCheckSwitch.setOnCheckedChangeListener { _, isChecked ->
            requireDrivingCheck = isChecked
            settings.edit { putBoolean(KEY_REQUIRE_DRIVING_CHECK, isChecked) }
            renderDriverState()
            if (isChecked) {
                ensureDrivingStatusMonitorStarted()
            }
        }

        binding.roadDetectionSwitch.isChecked = isRoadDetectionEnabled
        binding.roadDetectionSwitch.setOnCheckedChangeListener { _, isChecked ->
            isRoadDetectionEnabled = isChecked
            settings.edit { putBoolean(KEY_ROAD_DETECTION_ENABLED, isChecked) }
            rearAnalyzer?.detectionEnabled = isChecked
            if (!isChecked) {
                binding.roadObjectOverlay.render(null)
            }
        }

        updateDrivingStatusUi()
        handleDriverState(DriverState.Initializing)

        if (hasCameraPermission()) {
            startCameraPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        ensureDrivingStatusMonitorStarted()
    }

    override fun onPause() {
        super.onPause()
        drivingStatusMonitor.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        releaseCamera()
        cameraExecutor.shutdown()
        toneGenerator?.release()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCameraPipeline() {
        handleDriverState(DriverState.Initializing)
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                cameraProvider = future.get()
                bindUseCases()
            } catch (exception: Exception) {
                handleDriverState(DriverState.Error(exception.localizedMessage ?: getString(R.string.status_error)))
            }
        }, mainThreadExecutor)
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        provider.unbindAll()
        frontFpsTracker.reset()
        rearFpsTracker.reset()
        binding.frontFpsText.text = getString(R.string.fps_placeholder)
        binding.rearFpsText.text = getString(R.string.fps_placeholder)

        bindFrontCamera(provider)
        bindRearCamera(provider)
    }

    private fun bindFrontCamera(provider: ProcessCameraProvider) {
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.frontPreviewView.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also { it.setSurfaceProvider(binding.frontPreviewView.surfaceProvider) }

        val detectorOptions = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .build()

        faceDetector?.close()
        faceDetector = FaceDetection.getClient(detectorOptions)

        val analysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        val analyzer = DriveSenseAnalyzer(
            context = applicationContext,
            detector = faceDetector!!,
            mainExecutor = mainThreadExecutor,
            onStateUpdated = ::handleDriverState,
            onFrameProcessed = ::onFrontFrameProcessed
        )
        frontAnalyzer = analyzer
        analysis.setAnalyzer(cameraExecutor, analyzer)

        provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
    }

    private fun bindRearCamera(provider: ProcessCameraProvider) {
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.rearPreviewView.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also { it.setSurfaceProvider(binding.rearPreviewView.surfaceProvider) }

        val analysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        val analyzer = RoadObjectAnalyzer(
            context = applicationContext,
            callbackExecutor = mainThreadExecutor,
            onDetectionsUpdated = ::onRoadDetectionsUpdated,
            onFrameProcessed = ::onRearFrameProcessed
        ).also { it.detectionEnabled = isRoadDetectionEnabled }

        try {
            analysis.setAnalyzer(cameraExecutor, analyzer)
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            rearAnalyzer = analyzer
            binding.roadDetectionSwitch.isEnabled = true
        } catch (exception: Exception) {
            analysis.clearAnalyzer()
            analyzer.close()
            rearAnalyzer = null
            binding.roadDetectionSwitch.isEnabled = false
            if (isRoadDetectionEnabled) {
                isRoadDetectionEnabled = false
                binding.roadDetectionSwitch.isChecked = false
                settings.edit { putBoolean(KEY_ROAD_DETECTION_ENABLED, false) }
            }
            binding.roadObjectOverlay.render(null)
            binding.rearFpsText.text = getString(R.string.fps_placeholder)
        }
    }

    private fun onFrontFrameProcessed() {
        val fps = frontFpsTracker.tick()
        mainThreadExecutor.execute { updateFpsLabel(binding.frontFpsText, fps) }
    }

    private fun onRearFrameProcessed() {
        val fps = rearFpsTracker.tick()
        mainThreadExecutor.execute { updateFpsLabel(binding.rearFpsText, fps) }
    }

    private fun updateFpsLabel(textView: TextView, fps: Double) {
        val text = if (fps <= 0.5) {
            getString(R.string.fps_placeholder)
        } else {
            getString(R.string.fps_value, fps)
        }
        textView.text = text
    }

    private fun handleDriverState(state: DriverState) {
        latestDriverState = state
        renderDriverState()
    }

    private fun renderDriverState() {
        val state = latestDriverState
        when (state) {
            DriverState.Initializing -> {
                setStatus(getString(R.string.status_initializing), R.color.white)
                return
            }
            is DriverState.Error -> {
                setStatus(state.reason, R.color.status_drowsy)
                binding.lastEventText.visibility = View.GONE
                return
            }
            else -> Unit
        }

        if (requireDrivingCheck) {
            when (drivingDetectionStatus) {
                DrivingDetectionStatus.DRIVING -> Unit
                DrivingDetectionStatus.UNAVAILABLE -> {
                    setStatus(getString(R.string.status_driving_detection_unavailable), R.color.status_warning)
                    binding.lastEventText.visibility = View.GONE
                    return
                }
                DrivingDetectionStatus.UNKNOWN, DrivingDetectionStatus.STATIONARY -> {
                    setStatus(getString(R.string.status_waiting_for_driving), R.color.status_warning)
                    binding.lastEventText.visibility = View.GONE
                    return
                }
            }
        }

        when (state) {
            DriverState.NoFace -> setStatus(getString(R.string.status_searching), R.color.status_warning)
            DriverState.Attentive -> setStatus(getString(R.string.status_attentive), R.color.status_attentive)
            is DriverState.Drowsy -> {
                val secondsClosed = state.closedDurationMs / 1000f
                val message = getString(R.string.status_drowsy) + String.format(Locale.US, " (%.1fs)", secondsClosed)
                setStatus(message, R.color.status_drowsy)
                triggerDriverAlert()
            }
            else -> Unit
        }
    }

    private fun setStatus(message: CharSequence, colorResId: Int) {
        binding.statusText.text = message
        binding.statusText.setTextColor(ContextCompat.getColor(this, colorResId))
    }

    private fun triggerDriverAlert() {
        emitAlert { formattedTime -> getString(R.string.event_last_alert, formattedTime) }
    }

    private fun maybeTriggerRoadAlert(reason: String) {
        emitAlert { formattedTime -> getString(R.string.event_road_alert, reason, formattedTime) }
    }

    private fun emitAlert(messageFactory: (String) -> String) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastAlertTimestamp < ALERT_COOLDOWN_MS) {
            return
        }
        lastAlertTimestamp = now
        ensureToneGenerator().startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, ALERT_TONE_DURATION_MS.toInt())
        vibrate()
        val formattedTime = DateFormat.getTimeInstance(DateFormat.SHORT).format(Date())
        binding.lastEventText.visibility = View.VISIBLE
        binding.lastEventText.text = messageFactory(formattedTime)
    }

    private fun onRoadDetectionsUpdated(result: RoadObjectDetectionResult?) {
        binding.roadObjectOverlay.render(result)
        if (!isRoadDetectionEnabled || result == null) {
            return
        }

        val reasonRes = when {
            result.detections.any { it.category == RoadObjectCategory.ANIMAL } -> R.string.alert_reason_animal
            result.detections.any { it.category == RoadObjectCategory.PEDESTRIAN } -> R.string.alert_reason_pedestrian
            else -> null
        } ?: return

        maybeTriggerRoadAlert(getString(reasonRes))
    }

    private fun ensureToneGenerator(): ToneGenerator {
        if (toneGenerator == null) {
            toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, TONE_VOLUME)
        }
        return toneGenerator!!
    }

    private fun onDrivingStatusChanged(status: DrivingDetectionStatus) {
        drivingDetectionStatus = status
        updateDrivingStatusUi()
        renderDriverState()
    }

    private fun ensureDrivingStatusMonitorStarted() {
        if (!DrivingStatusMonitor.hasActivityRecognitionPermission(this)) {
            if (requireDrivingCheck && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                activityRecognitionPermissionLauncher.launch(Manifest.permission.ACTIVITY_RECOGNITION)
                return
            }
        }
        drivingStatusMonitor.start()
    }

    private fun updateDrivingStatusUi() {
        val text = when (drivingDetectionStatus) {
            DrivingDetectionStatus.DRIVING -> getString(R.string.status_driving_detected)
            DrivingDetectionStatus.STATIONARY -> getString(R.string.status_not_moving)
            DrivingDetectionStatus.UNAVAILABLE -> getString(R.string.status_driving_detection_unavailable)
            DrivingDetectionStatus.UNKNOWN -> getString(R.string.status_driving_unknown)
        }
        binding.drivingStatusText.text = text
    }

    private fun vibrate() {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(VIBRATION_DURATION_MS, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(VIBRATION_DURATION_MS)
        }
    }

    private fun releaseCamera() {
        cameraProvider?.unbindAll()
        frontAnalyzer?.close()
        frontAnalyzer = null
        rearAnalyzer?.close()
        rearAnalyzer = null
        faceDetector?.close()
        faceDetector = null
    }

    companion object {
        private const val ALERT_COOLDOWN_MS = 5000L
        private const val ALERT_TONE_DURATION_MS = 800L
        private const val VIBRATION_DURATION_MS = 500L
        private const val TONE_VOLUME = 100
        private const val PREFS_NAME = "drivesense_settings"
        private const val KEY_REQUIRE_DRIVING_CHECK = "require_driving_check"
        private const val KEY_ROAD_DETECTION_ENABLED = "road_detection_enabled"
    }
}
