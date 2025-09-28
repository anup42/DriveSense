package com.drivesense.drivesense

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.Surface
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.Lifecycle.Event
import com.drivesense.drivesense.databinding.ActivityMainBinding
import com.drivesense.drivesense.ui.DetectionOverlayView.RoadObjectDetection
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
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
    private var analyzer: DriveSenseAnalyzer? = null
    private var toneGenerator: ToneGenerator? = null
    private var roadObjectDetector: ObjectDetector? = null
    private var roadObjectAnalyzer: RoadObjectAnalyzer? = null
    private var lastAlertTimestamp: Long = 0L
    private lateinit var drivingStatusMonitor: DrivingStatusMonitor
    private var drivingDetectionStatus: DrivingDetectionStatus = DrivingDetectionStatus.UNKNOWN
    private var requireDrivingCheck: Boolean = true
    private var roadDetectionEnabled: Boolean = false
    private var latestDriverState: DriverState = DriverState.Initializing
    private val frontCameraLifecycleOwner = ManualLifecycleOwner()
    private val rearCameraLifecycleOwner = ManualLifecycleOwner()
    private var supportsConcurrentCameras: Boolean = false
    private var suppressRoadDetectionSwitchChange = false

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

        frontCameraLifecycleOwner.onCreate()
        rearCameraLifecycleOwner.onCreate()

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.lastEventText.visibility = View.GONE
        requireDrivingCheck = settings.getBoolean(KEY_REQUIRE_DRIVING_CHECK, true)
        roadDetectionEnabled = settings.getBoolean(KEY_ROAD_DETECTION_ENABLED, false)
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
        updateDrivingStatusUi()

        binding.roadDetectionSwitch.isChecked = roadDetectionEnabled
        binding.roadDetectionSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (suppressRoadDetectionSwitchChange) {
                return@setOnCheckedChangeListener
            }
            if (!supportsConcurrentCameras && isChecked) {
                setRoadDetectionSwitchChecked(false)
                return@setOnCheckedChangeListener
            }
            roadDetectionEnabled = isChecked
            settings.edit { putBoolean(KEY_ROAD_DETECTION_ENABLED, isChecked) }
            if (!isChecked) {
                binding.rearOverlay.clearDetections()
            }
            bindUseCases()
        }

        handleDriverState(DriverState.Initializing)

        if (hasCameraPermission()) {
            startCameraPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onStart() {
        super.onStart()
        frontCameraLifecycleOwner.onStart()
        rearCameraLifecycleOwner.onStart()
    }

    override fun onResume() {
        super.onResume()
        frontCameraLifecycleOwner.onResume()
        rearCameraLifecycleOwner.onResume()
        ensureDrivingStatusMonitorStarted()
    }

    override fun onPause() {
        super.onPause()
        frontCameraLifecycleOwner.onPause()
        rearCameraLifecycleOwner.onPause()
        drivingStatusMonitor.stop()
    }

    override fun onStop() {
        super.onStop()
        frontCameraLifecycleOwner.onStop()
        rearCameraLifecycleOwner.onStop()
    }

    override fun onDestroy() {
        super.onDestroy()
        releaseCamera()
        cameraExecutor.shutdown()
        toneGenerator?.release()
        rearCameraLifecycleOwner.onDestroy()
        frontCameraLifecycleOwner.onDestroy()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCameraPipeline() {
        handleDriverState(DriverState.Initializing)
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                supportsConcurrentCameras = provider.isConcurrentCameraModeSupportedCompat()
                updateConcurrentCameraAvailability()
                bindUseCases()
            } catch (exception: Exception) {
                handleDriverState(DriverState.Error(exception.localizedMessage ?: getString(R.string.status_error)))
            }
        }, mainThreadExecutor)
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        val frontPreview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.frontViewFinder.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also { it.setSurfaceProvider(binding.frontViewFinder.surfaceProvider) }

        val detectorOptions = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .build()

        faceDetector?.close()
        faceDetector = FaceDetection.getClient(detectorOptions)

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analyzer = DriveSenseAnalyzer(
            context = applicationContext,
            detector = faceDetector!!,
            mainExecutor = mainThreadExecutor,
            onStateUpdated = ::handleDriverState
        )
        imageAnalysis.setAnalyzer(cameraExecutor, analyzer!!)

        val frontUseCases = arrayOf<UseCase>(frontPreview, imageAnalysis)
        try {
            provider.bindToLifecycle(frontCameraLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, *frontUseCases)
        } catch (exception: Exception) {
            handleDriverState(DriverState.Error(exception.localizedMessage ?: getString(R.string.status_error)))
        }

        if (!supportsConcurrentCameras) {
            roadObjectAnalyzer = null
            releaseRoadObjectDetector()
            binding.rearOverlay.clearDetections()
            updateRearCameraUiVisibility(false)
            return
        }

        updateRearCameraUiVisibility(true)

        val rearPreview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.rearViewFinder.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also { it.setSurfaceProvider(binding.rearViewFinder.surfaceProvider) }

        val rearUseCases = mutableListOf<UseCase>(rearPreview)

        if (roadDetectionEnabled) {
            ensureRoadObjectDetector()
            val rearImageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            roadObjectAnalyzer = RoadObjectAnalyzer(
                detector = roadObjectDetector!!,
                mainExecutor = mainThreadExecutor,
                onDetectionsUpdated = ::onRoadDetectionsUpdated
            )
            rearImageAnalysis.setAnalyzer(cameraExecutor, roadObjectAnalyzer!!)
            rearUseCases.add(rearImageAnalysis)
        } else {
            roadObjectAnalyzer = null
            releaseRoadObjectDetector()
            binding.rearOverlay.clearDetections()
        }

        val rearUseCasesArray = rearUseCases.toTypedArray()
        var rearCameraBound = false
        var roadDetectionFallbackAttempted = false
        try {
            provider.bindToLifecycle(rearCameraLifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, *rearUseCasesArray)
            rearCameraBound = true
        } catch (exception: Exception) {
            Log.w(TAG, "Failed to bind rear camera use cases", exception)
            if (roadDetectionEnabled && rearUseCases.size > 1) {
                roadDetectionFallbackAttempted = true
                binding.rearOverlay.clearDetections()
                roadObjectAnalyzer = null
                releaseRoadObjectDetector()
                try {
                    provider.bindToLifecycle(rearCameraLifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, rearPreview)
                    rearCameraBound = true
                } catch (fallbackException: Exception) {
                    Log.e(TAG, "Failed to bind rear camera preview only fallback", fallbackException)
                }
            }
            if (!rearCameraBound && !roadDetectionFallbackAttempted) {
                binding.rearOverlay.clearDetections()
                releaseRoadObjectDetector()
            }
        }
    }

    private fun onRoadDetectionsUpdated(detections: List<RoadObjectDetection>) {
        if (roadDetectionEnabled) {
            binding.rearOverlay.updateDetections(detections)
        }
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
                triggerAlerts()
            }
            else -> Unit
        }
    }

    private fun setStatus(message: CharSequence, colorResId: Int) {
        binding.statusText.text = message
        binding.statusText.setTextColor(ContextCompat.getColor(this, colorResId))
    }

    private fun triggerAlerts() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastAlertTimestamp < ALERT_COOLDOWN_MS) {
            return
        }
        lastAlertTimestamp = now

        ensureToneGenerator().startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, ALERT_TONE_DURATION_MS.toInt())
        vibrate()
        updateLastAlertLabel()
    }

    private fun updateLastAlertLabel() {
        val formattedTime = DateFormat.getTimeInstance(DateFormat.SHORT).format(Date())
        binding.lastEventText.visibility = View.VISIBLE
        binding.lastEventText.text = getString(R.string.event_last_alert, formattedTime)
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
        analyzer?.close()
        analyzer = null
        faceDetector?.close()
        faceDetector = null
        roadObjectAnalyzer = null
        releaseRoadObjectDetector()
        binding.rearOverlay.clearDetections()
    }

    private fun ProcessCameraProvider.isConcurrentCameraModeSupportedCompat(): Boolean {
        return try {
            val method = ProcessCameraProvider::class.java.getMethod(
                "isConcurrentCameraModeSupported",
                CameraSelector::class.java,
                CameraSelector::class.java
            )
            method.invoke(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                CameraSelector.DEFAULT_BACK_CAMERA
            ) as Boolean
        } catch (error: NoSuchMethodException) {
            Log.w(TAG, "Concurrent camera mode support check unavailable via CameraX API", error)
            checkConcurrentCameraSupportViaCameraManager()
        } catch (error: NoSuchMethodError) {
            Log.w(TAG, "Concurrent camera mode support check unavailable via CameraX API", error)
            checkConcurrentCameraSupportViaCameraManager()
        } catch (exception: Exception) {
            Log.w(TAG, "Unable to query concurrent camera mode support", exception)
            false
        }
    }

    private fun ProcessCameraProvider.checkConcurrentCameraSupportViaCameraManager(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            Log.i(TAG, "Concurrent camera mode requires API 30+, current API: ${Build.VERSION.SDK_INT}")
            return false
        }

        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        if (cameraManager == null) {
            Log.w(TAG, "CameraManager unavailable, cannot check concurrent camera support")
            return false
        }

        val concurrentCameraIds = try {
            cameraManager.concurrentCameraIds
        } catch (error: SecurityException) {
            Log.w(TAG, "Unable to query concurrent camera ids due to missing permission", error)
            return false
        } catch (error: Throwable) {
            Log.w(TAG, "Unexpected error while querying concurrent camera ids", error)
            return false
        }

        if (concurrentCameraIds.isNullOrEmpty()) {
            Log.i(TAG, "Device does not report any concurrent camera combinations")
            return false
        }

        val cameraInfos = availableCameraInfos
        val frontCameraIds = cameraInfos.mapNotNull { info ->
            val cameraInfo = Camera2CameraInfo.from(info)
            val lensFacing = cameraInfo.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
            if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                cameraInfo.cameraId
            } else {
                null
            }
        }.toSet()

        val backCameraIds = cameraInfos.mapNotNull { info ->
            val cameraInfo = Camera2CameraInfo.from(info)
            val lensFacing = cameraInfo.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
            if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
                cameraInfo.cameraId
            } else {
                null
            }
        }.toSet()

        if (frontCameraIds.isEmpty() || backCameraIds.isEmpty()) {
            Log.i(TAG, "Missing required front or back cameras for concurrent mode")
            return false
        }

        val supportsFrontBackCombo = concurrentCameraIds.any { combo ->
            combo.any { it in frontCameraIds } && combo.any { it in backCameraIds }
        }

        if (!supportsFrontBackCombo) {
            Log.i(TAG, "No concurrent camera combination includes both front and back cameras")
        }

        return supportsFrontBackCombo
    }

    private fun ensureRoadObjectDetector() {
        if (roadObjectDetector != null) return
        val options = ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
        roadObjectDetector = ObjectDetection.getClient(options)
    }

    private fun releaseRoadObjectDetector() {
        roadObjectDetector?.close()
        roadObjectDetector = null
    }

    private fun updateConcurrentCameraAvailability() {
        binding.roadDetectionSwitch.isEnabled = supportsConcurrentCameras
        if (!supportsConcurrentCameras && roadDetectionEnabled) {
            roadDetectionEnabled = false
            settings.edit { putBoolean(KEY_ROAD_DETECTION_ENABLED, false) }
            setRoadDetectionSwitchChecked(false)
        }
    }

    private fun setRoadDetectionSwitchChecked(checked: Boolean) {
        if (binding.roadDetectionSwitch.isChecked == checked) return
        suppressRoadDetectionSwitchChange = true
        binding.roadDetectionSwitch.isChecked = checked
        suppressRoadDetectionSwitchChange = false
    }

    private fun updateRearCameraUiVisibility(showRearPreview: Boolean) {
        binding.rearPreviewContainer.visibility = if (showRearPreview) View.VISIBLE else View.GONE
        binding.midGuideline.setGuidelinePercent(if (showRearPreview) 0.5f else 1f)
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val ALERT_COOLDOWN_MS = 5000L
        private const val ALERT_TONE_DURATION_MS = 800L
        private const val VIBRATION_DURATION_MS = 500L
        private const val TONE_VOLUME = 100
        private const val PREFS_NAME = "drivesense_settings"
        private const val KEY_REQUIRE_DRIVING_CHECK = "require_driving_check"
        private const val KEY_ROAD_DETECTION_ENABLED = "road_detection_enabled"
    }
}

private class ManualLifecycleOwner : LifecycleOwner {
    private val lifecycleRegistry = LifecycleRegistry(this)

    override val lifecycle: Lifecycle
        get() = lifecycleRegistry

    fun onCreate() {
        lifecycleRegistry.handleLifecycleEvent(Event.ON_CREATE)
    }

    fun onStart() {
        lifecycleRegistry.handleLifecycleEvent(Event.ON_START)
    }

    fun onResume() {
        lifecycleRegistry.handleLifecycleEvent(Event.ON_RESUME)
    }

    fun onPause() {
        lifecycleRegistry.handleLifecycleEvent(Event.ON_PAUSE)
    }

    fun onStop() {
        lifecycleRegistry.handleLifecycleEvent(Event.ON_STOP)
    }

    fun onDestroy() {
        lifecycleRegistry.handleLifecycleEvent(Event.ON_DESTROY)
    }
}







