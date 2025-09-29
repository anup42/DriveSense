package com.drivesense.drivesense.camera2

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.view.Surface
import androidx.annotation.RequiresPermission
import androidx.core.content.ContextCompat

class FrontCamera2Pipeline(
    private val context: Context,
    private val targetSize: Size = Size(320, 240),
    private val maxImages: Int = 2,
    private val desiredFps: Range<Int> = Range(15, 30),
    private val onFrame: (image: Image, rotationDegrees: Int) -> Unit
) {

    private val cameraManager by lazy {
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null

    private var cameraDevice: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var cameraId: String? = null
    private var characteristics: CameraCharacteristics? = null

    private var processing = false

    fun start(displayRotation: Int = Surface.ROTATION_0) {
        if (!hasPermission()) return

        stop()

        startBgThread()

        cameraId = findFrontCameraId() ?: return
        characteristics = cameraManager.getCameraCharacteristics(cameraId!!)
        imageReader = ImageReader.newInstance(
            targetSize.width,
            targetSize.height,
            ImageFormat.YUV_420_888,
            maxImages
        ).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                if (processing) {
                    image.close()
                    return@setOnImageAvailableListener
                }
                processing = true
                val rotationDegrees = computeRotationDegrees(displayRotation)
                try {
                    onFrame(image, rotationDegrees)
                } catch (_: Throwable) {
                    image.close()
                    processing = false
                }
            }, bgHandler)
        }

        @Suppress("MissingPermission")
        openCamera(displayRotation)
    }

    fun stop() {
        try {
            session?.close()
        } catch (_: Throwable) {
        }
        session = null

        try {
            cameraDevice?.close()
        } catch (_: Throwable) {
        }
        cameraDevice = null

        try {
            imageReader?.close()
        } catch (_: Throwable) {
        }
        imageReader = null

        stopBgThread()
        processing = false
    }

    private fun hasPermission(): Boolean {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    @RequiresPermission(Manifest.permission.CAMERA)
    private fun openCamera(displayRotation: Int) {
        val id = cameraId ?: return
        cameraManager.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                createSession(displayRotation)
            }

            override fun onDisconnected(device: CameraDevice) {
                device.close()
                cameraDevice = null
            }

            override fun onError(device: CameraDevice, error: Int) {
                device.close()
                cameraDevice = null
            }
        }, bgHandler)
    }

    private fun createSession(displayRotation: Int) {
        val device = cameraDevice ?: return
        val readerSurface = imageReader?.surface ?: return

        val targets = listOf(readerSurface)
        device.createCaptureSession(
            targets,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    this@FrontCamera2Pipeline.session = session

                    val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                        addTarget(readerSurface)
                        set(
                            CaptureRequest.CONTROL_AF_MODE,
                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                        )
                        set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, desiredFps)
                        set(
                            CaptureRequest.NOISE_REDUCTION_MODE,
                            CaptureRequest.NOISE_REDUCTION_MODE_FAST
                        )
                    }.build()

                    session.setRepeatingRequest(request, null, bgHandler)
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                }
            },
            bgHandler
        )
    }

    private fun findFrontCameraId(): String? {
        for (id in cameraManager.cameraIdList) {
            val chars = cameraManager.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                return id
            }
        }
        return null
    }

    private fun computeRotationDegrees(deviceRotation: Int): Int {
        val sensorOrientation = characteristics?.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val surfaceDegrees = when (deviceRotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensorOrientation + surfaceDegrees) % 360
    }

    private fun startBgThread() {
        val t = HandlerThread("FrontCam2Bg")
        t.start()
        bgThread = t
        bgHandler = Handler(t.looper)
    }

    private fun stopBgThread() {
        bgThread?.quitSafely()
        try {
            bgThread?.join()
        } catch (_: InterruptedException) {
        }
        bgThread = null
        bgHandler = null
    }

    fun markFrameDone(image: Image) {
        try {
            image.close()
        } catch (_: Throwable) {
        }
        processing = false
    }
}
