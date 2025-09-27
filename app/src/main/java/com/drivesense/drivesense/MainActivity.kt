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
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
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
    private var analyzer: DriveSenseAnalyzer? = null
    private var toneGenerator: ToneGenerator? = null
    private var lastAlertTimestamp: Long = 0L

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCameraPipeline()
            } else {
                handleDriverState(DriverState.Error(getString(R.string.status_permission_denied)))
            }
        }

    private val mainThreadExecutor: Executor by lazy { ContextCompat.getMainExecutor(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.lastEventText.visibility = View.GONE
        handleDriverState(DriverState.Initializing)

        if (hasCameraPermission()) {
            startCameraPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
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

        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display?.rotation ?: Surface.ROTATION_0)
            .build()
            .also { it.setSurfaceProvider(binding.viewFinder.surfaceProvider) }

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

        provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalysis)
    }

    private fun handleDriverState(state: DriverState) {
        when (state) {
            DriverState.Initializing -> setStatus(getString(R.string.status_initializing), R.color.white)
            DriverState.NoFace -> setStatus(getString(R.string.status_searching), R.color.status_warning)
            DriverState.Attentive -> setStatus(getString(R.string.status_attentive), R.color.status_attentive)
            is DriverState.Drowsy -> {
                val secondsClosed = state.closedDurationMs / 1000f
                val message = getString(R.string.status_drowsy) + String.format(Locale.US, " (%.1fs)", secondsClosed)
                setStatus(message, R.color.status_drowsy)
                triggerAlerts()
            }
            is DriverState.Error -> {
                setStatus(state.reason, R.color.status_drowsy)
                binding.lastEventText.visibility = View.GONE
            }
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
    }

    companion object {
        private const val ALERT_COOLDOWN_MS = 5000L
        private const val ALERT_TONE_DURATION_MS = 800L
        private const val VIBRATION_DURATION_MS = 500L
        private const val TONE_VOLUME = 100
    }
}







