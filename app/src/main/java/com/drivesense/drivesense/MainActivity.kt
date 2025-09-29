package com.drivesense.drivesense

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
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
import android.util.Size
import android.view.Surface
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.core.view.doOnLayout
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
    private var showFrontPreview: Boolean = true
    private var showRearPreview: Boolean = true
    private var latestDriverState: DriverState = DriverState.Initializing
    private var supportsConcurrentCameras: Boolean = false
    private var suppressRoadDetectionSwitchChange = false
    private var suppressFrontPreviewSwitchChange = false
    private var suppressRearPreviewSwitchChange = false
    private var concurrentFrontCameraId: String? = null
    private var concurrentBackCameraId: String? = null
    private var frontPreviewResolution: Size? = null
    private var rearPreviewResolution: Size? = null
    private var frontPreviewGuidelinePercent: Float? = null
    private var rearPreviewAvailable: Boolean = false

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

        binding.frontViewFinder.implementationMode =
            PreviewView.ImplementationMode.PERFORMANCE
        binding.frontViewFinder.scaleType =
            PreviewView.ScaleType.FIT_CENTER
        binding.rearViewFinder.implementationMode =
            PreviewView.ImplementationMode.PERFORMANCE
        binding.rearViewFinder.scaleType =
            PreviewView.ScaleType.FIT_CENTER

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.lastEventText.visibility = View.GONE
        requireDrivingCheck = settings.getBoolean(KEY_REQUIRE_DRIVING_CHECK, true)
        roadDetectionEnabled = settings.getBoolean(KEY_ROAD_DETECTION_ENABLED, false)
        showFrontPreview = settings.getBoolean(KEY_SHOW_FRONT_PREVIEW, true)
        showRearPreview = settings.getBoolean(KEY_SHOW_REAR_PREVIEW, true)
        drivingStatusMonitor = DrivingStatusMonitor(
            context = this,
            callbackExecutor = mainThreadExecutor,
            onStatusChanged = ::onDrivingStatusChanged
        )

        setFrontPreviewSwitchChecked(showFrontPreview)
        binding.frontPreviewSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (suppressFrontPreviewSwitchChange) {
                return@setOnCheckedChangeListener
            }
            showFrontPreview = isChecked
            settings.edit { putBoolean(KEY_SHOW_FRONT_PREVIEW, isChecked) }
            applyFrontPreviewVisibility()
        }

        setRearPreviewSwitchChecked(showRearPreview)
        binding.rearPreviewSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (suppressRearPreviewSwitchChange) {
                return@setOnCheckedChangeListener
            }
            showRearPreview = isChecked
            settings.edit { putBoolean(KEY_SHOW_REAR_PREVIEW, isChecked) }
            applyRearPreviewVisibility()
        }

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

        applyFrontPreviewVisibility()
        applyRearPreviewVisibility()

        handleDriverState(DriverState.Initializing)

        if (hasCameraPermission()) {
            startCameraPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onStart() {
        super.onStart()
    }

    override fun onResume() {
        super.onResume()
        ensureDrivingStatusMonitorStarted()
        if (hasCameraPermission()) {
            bindUseCases()
        }
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
                val provider = future.get()
                cameraProvider = provider
                supportsConcurrentCameras = provider.isConcurrentCameraModeSupportedCompat()
                Log.i(TAG, "CameraX concurrent camera support reported as $supportsConcurrentCameras")
                updateConcurrentCameraAvailability()
                bindUseCases()
            } catch (exception: Exception) {
                handleDriverState(DriverState.Error(exception.localizedMessage ?: getString(R.string.status_error)))
            }
        }, mainThreadExecutor)
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        val rearAvailable = supportsConcurrentCameras && roadDetectionEnabled
        setRearPreviewAvailability(rearAvailable)
        applyFrontPreviewVisibility()
        runWhenPreviewViewsReady {
            bindUseCasesInternal(provider)
        }
    }

    private fun runWhenPreviewViewsReady(action: () -> Unit) {
        val frontRequired = !(supportsConcurrentCameras && roadDetectionEnabled)
        val frontReady = !frontRequired || isPreviewViewReady(binding.frontViewFinder)
        val rearRequired = supportsConcurrentCameras && roadDetectionEnabled && showRearPreview
        val rearReady = !rearRequired || isPreviewViewReady(binding.rearViewFinder)
        if (frontReady && rearReady) {
            action()
            return
        }
        binding.frontViewFinder.post {
            val frontPostReady = !frontRequired || isPreviewViewReady(binding.frontViewFinder)
            val rearPostReady = !rearRequired || isPreviewViewReady(binding.rearViewFinder)
            if (frontPostReady && rearPostReady) {
                action()
            } else if (rearRequired) {
                binding.rearViewFinder.post {
                    val finalFrontReady = !frontRequired || isPreviewViewReady(binding.frontViewFinder)
                    if (finalFrontReady && isPreviewViewReady(binding.rearViewFinder)) {
                        action()
                    }
                }
            }
        }
    }

    private fun isPreviewViewReady(view: View): Boolean {
        return view.width > 0 && view.height > 0
    }

    private fun bindUseCasesInternal(provider: ProcessCameraProvider) {
        provider.unbindAll()

        val detectorOptions = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .build()

        faceDetector?.close()
        faceDetector = FaceDetection.getClient(detectorOptions)

        analyzer?.close()
        analyzer = DriveSenseAnalyzer(
            context = applicationContext,
            detector = faceDetector!!,
            mainExecutor = mainThreadExecutor,
            onStateUpdated = ::handleDriverState
        )

        if (supportsConcurrentCameras && roadDetectionEnabled) {
            applyFrontPreviewVisibility()
            applyRearPreviewVisibility()
            if (bindFrontAndRearCameras(provider)) {
                return
            }

            Log.w(TAG, "Falling back to single-camera mode after concurrent bind failure")
            roadDetectionEnabled = false
            settings.edit { putBoolean(KEY_ROAD_DETECTION_ENABLED, false) }
            setRoadDetectionSwitchChecked(false)
            setRearPreviewAvailability(false)
        }

        applyFrontPreviewVisibility()
        setRearPreviewAvailability(false)
        bindFrontCameraOnly(provider)
    }

    private fun bindFrontCameraOnly(provider: ProcessCameraProvider) {
        val analyzerInstance = analyzer ?: run {
            Log.w(TAG, "Analyzer unavailable; cannot bind front camera")
            return
        }

        roadObjectAnalyzer = null
        releaseRoadObjectDetector()
        binding.rearOverlay.clearDetections()

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(ANALYSIS_TARGET_RESOLUTION)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageAnalysis.setAnalyzer(cameraExecutor, analyzerInstance)

        val frontSurfaceProvider = binding.frontViewFinder.surfaceProvider
        val configuredFrontResolution = frontPreviewResolution
            ?: findHighestPreviewResolution(CameraCharacteristics.LENS_FACING_FRONT)
                ?.also { frontPreviewResolution = it }
        val frontPreviewBuilder = Preview.Builder()
        configuredFrontResolution?.let { frontPreviewBuilder.setTargetResolution(it) }
        val frontPreview = frontPreviewBuilder
            .setTargetRotation(binding.frontViewFinder.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also {
                it.setSurfaceProvider { request ->
                    Log.d(TAG, "Front surface requested: ${request.resolution}")
                    frontPreviewResolution = request.resolution
                    updateFrontPreviewAspect(request.resolution)
                    frontSurfaceProvider.onSurfaceRequested(request)
                }
            }

        val frontGroup = UseCaseGroup.Builder()
            .addUseCase(frontPreview)
            .addUseCase(imageAnalysis)
            .build()

        try {
            val frontCamera = provider.bindToLifecycle(this, getFrontCameraSelector(), frontGroup)
            logBound(frontCamera, "FRONT")
            observeFrontCameraState(frontCamera)
            observeRearCameraState(null)
        } catch (exception: Exception) {
            handleDriverState(DriverState.Error(exception.localizedMessage ?: getString(R.string.status_error)))
        }
    }

    private fun bindFrontAndRearCameras(provider: ProcessCameraProvider): Boolean {
        val analyzerInstance = analyzer ?: run {
            Log.w(TAG, "Analyzer unavailable; cannot bind front camera")
            return false
        }

        ensureRoadObjectDetector()
        val detector = roadObjectDetector ?: run {
            Log.w(TAG, "Road object detector unavailable; cannot enable road detection")
            roadObjectAnalyzer = null
            binding.rearOverlay.clearDetections()
            return false
        }

        val frontSurfaceProvider = binding.frontViewFinder.surfaceProvider
        val configuredFrontResolution = frontPreviewResolution
            ?: findHighestPreviewResolution(CameraCharacteristics.LENS_FACING_FRONT)
                ?.also { frontPreviewResolution = it }
        val frontPreviewBuilder = Preview.Builder()
        configuredFrontResolution?.let { frontPreviewBuilder.setTargetResolution(it) }
        val frontPreview = frontPreviewBuilder
            .setTargetRotation(binding.frontViewFinder.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also {
                it.setSurfaceProvider { request ->
                    Log.d(TAG, "Front surface requested: ${request.resolution}")
                    frontPreviewResolution = request.resolution
                    updateFrontPreviewAspect(request.resolution)
                    frontSurfaceProvider.onSurfaceRequested(request)
                }
            }

        val frontImageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(ANALYSIS_TARGET_RESOLUTION)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        frontImageAnalysis.setAnalyzer(cameraExecutor, analyzerInstance)

        val frontGroup = UseCaseGroup.Builder()
            .addUseCase(frontPreview)
            .addUseCase(frontImageAnalysis)
            .build()

        val rearSurfaceProvider = binding.rearViewFinder.surfaceProvider
        val configuredRearResolution = rearPreviewResolution
            ?: findHighestPreviewResolution(CameraCharacteristics.LENS_FACING_BACK)
                ?.also { rearPreviewResolution = it }
        val rearPreviewBuilder = Preview.Builder()
        configuredRearResolution?.let { rearPreviewBuilder.setTargetResolution(it) }
        val rearPreview = rearPreviewBuilder
            .setTargetRotation(binding.rearViewFinder.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also {
                it.setSurfaceProvider { request ->
                    Log.d(TAG, "Rear surface requested: ${request.resolution}")
                    rearPreviewResolution = request.resolution
                    rearSurfaceProvider.onSurfaceRequested(request)
                }
            }

        val rearImageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(ANALYSIS_TARGET_RESOLUTION)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        val roadAnalyzer = RoadObjectAnalyzer(
            detector = detector,
            mainExecutor = mainThreadExecutor,
            onDetectionsUpdated = ::onRoadDetectionsUpdated
        )
        roadObjectAnalyzer = roadAnalyzer
        rearImageAnalysis.setAnalyzer(cameraExecutor, roadAnalyzer)

        val rearGroup = UseCaseGroup.Builder()
            .addUseCase(rearPreview)
            .addUseCase(rearImageAnalysis)
            .build()

        return try {
            val frontCamera = provider.bindToLifecycle(this, getFrontCameraSelector(), frontGroup)
            val rearCamera = provider.bindToLifecycle(this, getRearCameraSelector(), rearGroup)
            logBound(frontCamera, "FRONT")
            logBound(rearCamera, "REAR")
            observeFrontCameraState(frontCamera)
            observeRearCameraState(rearCamera)
            true
        } catch (exception: Exception) {
            Log.e(TAG, "Failed to bind concurrent front and rear camera use cases", exception)
            provider.unbind(frontPreview, frontImageAnalysis, rearPreview, rearImageAnalysis)
            roadObjectAnalyzer = null
            releaseRoadObjectDetector()
            binding.rearOverlay.clearDetections()
            false
        }
    }

    private fun observeFrontCameraState(camera: Camera?) {
        camera?.cameraInfo?.cameraState?.observe(this) { state ->
            Log.d(TAG, "Front camera state: ${state.type} error=${state.error?.code}")
        }
    }

    private fun observeRearCameraState(camera: Camera?) {
        camera?.cameraInfo?.cameraState?.observe(this) { state ->
            Log.d(TAG, "Rear camera state: ${state.type} error=${state.error?.code}")
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
        concurrentFrontCameraId = null
        concurrentBackCameraId = null

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            Log.i(TAG, "Concurrent camera mode requires API 30+, current API: ${Build.VERSION.SDK_INT}")
            return false
        }

        val concurrentCameraInfos = availableConcurrentCameraInfos
        if (concurrentCameraInfos.isNotEmpty()) {
            Log.i(TAG, "CameraX reports ${concurrentCameraInfos.size} concurrent camera combo(s)")
            for (combo in concurrentCameraInfos) {
                var frontInfo: androidx.camera.core.CameraInfo? = null
                var backInfo: androidx.camera.core.CameraInfo? = null
                for (info in combo) {
                    val facing = Camera2CameraInfo.from(info)
                        .getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                    when (facing) {
                        CameraCharacteristics.LENS_FACING_FRONT -> if (frontInfo == null) frontInfo = info
                        CameraCharacteristics.LENS_FACING_BACK -> if (backInfo == null) backInfo = info
                    }
                }
                if (frontInfo != null && backInfo != null) {
                    concurrentFrontCameraId = Camera2CameraInfo.from(frontInfo!!).cameraId
                    concurrentBackCameraId = Camera2CameraInfo.from(backInfo!!).cameraId
                    Log.i(TAG, "Dual camera support confirmed via CameraX concurrent combos")
                    return true
                }
            }
            Log.i(TAG, "No CameraX concurrent combo includes both front and back cameras")
        }

        val packageManager = packageManager
        if (!packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_CONCURRENT)) {
            Log.i(TAG, "Device does not report FEATURE_CAMERA_CONCURRENT")
            return false
        }
        Log.i(TAG, "Device reports FEATURE_CAMERA_CONCURRENT")

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

        Log.i(
            TAG,
            "CameraManager reports ${concurrentCameraIds.size} concurrent camera combination(s): " +
                concurrentCameraIds.joinToString(prefix = "[", postfix = "]") { combo ->
                    combo.joinToString(prefix = "[", postfix = "]")
                }
        )

        val availableCameraIds = availableCameraInfos
            .map { info -> Camera2CameraInfo.from(info).cameraId }
            .toSet()

        Log.i(TAG, "CameraX reports available camera ids: $availableCameraIds")

        for (combo in concurrentCameraIds) {
            if (combo.size < 2) continue
            var frontId: String? = null
            var backId: String? = null
            for (cameraId in combo) {
                if (cameraId !in availableCameraIds) continue
                val lensFacing = try {
                    cameraManager.getCameraCharacteristics(cameraId)
                        .get(CameraCharacteristics.LENS_FACING)
                } catch (error: Exception) {
                    Log.w(TAG, "Unable to read lens facing for camera $cameraId", error)
                    null
                }
                Log.d(TAG, "Evaluating concurrent combo cameraId=$cameraId lensFacing=$lensFacing")
                when (lensFacing) {
                    CameraCharacteristics.LENS_FACING_FRONT -> if (frontId == null) frontId = cameraId
                    CameraCharacteristics.LENS_FACING_BACK -> if (backId == null) backId = cameraId
                }
            }
            if (frontId != null && backId != null) {
                concurrentFrontCameraId = frontId
                concurrentBackCameraId = backId
                Log.i(TAG, "Concurrent camera combo selected: front=$frontId back=$backId")
                Log.i(TAG, "Dual camera support confirmed via Camera2 concurrent combo")
                return true
            }
        }

        Log.i(TAG, "No concurrent camera combination includes both front and back cameras")
        return false
    }

    private fun getFrontCameraSelector(): CameraSelector {
        val provider = cameraProvider
        concurrentFrontCameraId?.let { id ->
            return CameraSelector.Builder()
                .addCameraFilter { infos ->
                    infos.filter { Camera2CameraInfo.from(it).cameraId == id }
                        .ifEmpty { infos }
                }
                .build()
        }

        val frontId = provider?.availableCameraInfos
            ?.map { Camera2CameraInfo.from(it) }
            ?.firstOrNull {
                it.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_FRONT
            }
            ?.cameraId

        return if (frontId != null) {
            CameraSelector.Builder()
                .addCameraFilter { infos ->
                    infos.filter { Camera2CameraInfo.from(it).cameraId == frontId }
                }
                .build()
        } else {
            CameraSelector.DEFAULT_FRONT_CAMERA
        }
    }

    private fun getRearCameraSelector(): CameraSelector {
        val provider = cameraProvider
        concurrentBackCameraId?.let { id ->
            return CameraSelector.Builder()
                .addCameraFilter { infos ->
                    infos.filter { Camera2CameraInfo.from(it).cameraId == id }
                        .ifEmpty { infos }
                }
                .build()
        }

        val backId = provider?.availableCameraInfos
            ?.map { Camera2CameraInfo.from(it) }
            ?.firstOrNull {
                it.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            }
            ?.cameraId

        return if (backId != null) {
            CameraSelector.Builder()
                .addCameraFilter { infos ->
                    infos.filter { Camera2CameraInfo.from(it).cameraId == backId }
                }
                .build()
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
    }

    private fun logBound(camera: Camera?, label: String) {
        camera ?: return
        val info = camera.cameraInfo
        val id = Camera2CameraInfo.from(info).cameraId
        val facing = Camera2CameraInfo.from(info)
            .getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
        Log.i(TAG, "$label bound to cameraId=$id facing=$facing")
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
        if (!supportsConcurrentCameras) {
            setRearPreviewAvailability(false)
        }
    }

    private fun setRoadDetectionSwitchChecked(checked: Boolean) {
        if (binding.roadDetectionSwitch.isChecked == checked) return
        suppressRoadDetectionSwitchChange = true
        binding.roadDetectionSwitch.isChecked = checked
        suppressRoadDetectionSwitchChange = false
    }

    private fun setFrontPreviewSwitchChecked(checked: Boolean) {
        if (binding.frontPreviewSwitch.isChecked == checked) return
        suppressFrontPreviewSwitchChange = true
        binding.frontPreviewSwitch.isChecked = checked
        suppressFrontPreviewSwitchChange = false
    }

    private fun setRearPreviewSwitchChecked(checked: Boolean) {
        if (binding.rearPreviewSwitch.isChecked == checked) return
        suppressRearPreviewSwitchChange = true
        binding.rearPreviewSwitch.isChecked = checked
        suppressRearPreviewSwitchChange = false
    }

    private fun setRearPreviewAvailability(available: Boolean) {
        rearPreviewAvailable = available
        binding.rearPreviewSwitch.isEnabled = available
        if (!available) {
            setRearPreviewSwitchChecked(false)
            binding.rearPreviewContainer.visibility = View.GONE
            applyFrontPreviewGuidelinePercent()
        } else {
            setRearPreviewSwitchChecked(showRearPreview)
            applyRearPreviewVisibility()
        }
    }

    private fun applyFrontPreviewVisibility() {
        binding.frontPreviewContainer.visibility =
            if (showFrontPreview) View.VISIBLE else View.INVISIBLE
        if (showFrontPreview) {
            if (!rearPreviewAvailable || !showRearPreview) {
                applyFrontPreviewGuidelinePercent()
            }
        } else {
            applyFrontPreviewGuidelinePercent()
        }
    }

    private fun applyRearPreviewVisibility() {
        if (!rearPreviewAvailable) {
            binding.rearPreviewContainer.visibility = View.GONE
            applyFrontPreviewGuidelinePercent()
            return
        }
        binding.rearPreviewContainer.visibility =
            if (showRearPreview) View.VISIBLE else View.INVISIBLE
        if (showRearPreview) {
            binding.midGuideline.setGuidelinePercent(0.5f)
        } else {
            applyFrontPreviewGuidelinePercent()
        }
    }

    private fun updateFrontPreviewAspect(resolution: Size) {
        if (resolution.width <= 0 || resolution.height <= 0) {
            return
        }
        val rootView = binding.root
        if (rootView.width == 0 || rootView.height == 0) {
            rootView.doOnLayout { updateFrontPreviewAspect(resolution) }
            return
        }
        val desiredHeight = rootView.width.toFloat() * resolution.height.toFloat() /
            resolution.width.toFloat()
        val percent = (desiredHeight / rootView.height.toFloat()).coerceIn(0f, 1f)
        frontPreviewGuidelinePercent = percent
        applyFrontPreviewGuidelinePercent()
    }

    private fun applyFrontPreviewGuidelinePercent() {
        if (binding.rearPreviewContainer.visibility == View.VISIBLE) {
            return
        }
        val percent = frontPreviewGuidelinePercent ?: 1f
        binding.midGuideline.setGuidelinePercent(percent)
    }

    private fun findHighestPreviewResolution(lensFacing: Int): Size? {
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as? CameraManager ?: return null
        var bestSize: Size? = null
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (facing != lensFacing) continue
            val configMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val sizes = configMap?.getOutputSizes(SurfaceTexture::class.java) ?: continue
            val candidate = sizes.maxByOrNull { size -> size.width.toLong() * size.height.toLong() }
            if (candidate != null) {
                val candidatePixels = candidate.width.toLong() * candidate.height.toLong()
                val currentPixels = bestSize?.let { it.width.toLong() * it.height.toLong() } ?: -1L
                if (candidatePixels > currentPixels) {
                    bestSize = candidate
                }
            }
        }
        return bestSize
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
        private const val KEY_SHOW_FRONT_PREVIEW = "show_front_preview"
        private const val KEY_SHOW_REAR_PREVIEW = "show_rear_preview"
        private val ANALYSIS_TARGET_RESOLUTION = Size(640, 480)
    }
}







