package com.drivesense.drivesense.camera2

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.annotation.MainThread
import androidx.annotation.WorkerThread
import kotlin.math.max

/**
 * Minimal Camera2 pipeline that opens the **front** camera and feeds YUV frames
 * to a callback for ML (no on-screen preview).
 *
 * Usage:
 *   val pipe = FrontCamera2Pipeline(context, Size(320, 240)) { image, rotation ->
 *       // Do ML Kit here, then call:
 *       pipeline.markFrameDone(image)
 *   }
 *   pipe.start(displayRotation)
 *   ...
 *   pipe.stop()
 */
class FrontCamera2Pipeline(
    private val context: Context,
    private val targetSize: Size,
    private val onFrame: (image: Image, rotationDegrees: Int) -> Unit
) {

    companion object {
        private const val TAG = "FrontCam2"
        private const val MAX_IMAGES = 2 // keep small to avoid backpressure
    }

    private val camMgr: CameraManager =
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var camera: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null

    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null

    private var cameraId: String? = null
    private var sensorOrientation: Int = 0
    private var isFrontFacing = true
    private var latestDisplayRotation: Int = Surface.ROTATION_0

    private var closed = false

    @MainThread
    fun start(displayRotation: Int) {
        latestDisplayRotation = displayRotation
        startThread()
        chooseFrontCamera()
        openCamera()
    }

    @MainThread
    fun stop() {
        closed = true
        try {
            captureSession?.close()
        } catch (_: Throwable) {}
        try {
            camera?.close()
        } catch (_: Throwable) {}
        captureSession = null
        camera = null

        imageReader?.close()
        imageReader = null

        stopThread()
    }

    /**
     * Call this **once** when you're done with the Image provided to onFrame.
     */
    fun markFrameDone(image: Image) {
        try { image.close() } catch (_: Throwable) {}
    }

    // ---------- internals ----------

    private fun startThread() {
        if (bgThread != null) return
        bgThread = HandlerThread("FrontCam2Thread").apply { start() }
        bgHandler = Handler(bgThread!!.looper)
    }

    private fun stopThread() {
        bgThread?.quitSafely()
        try {
            bgThread?.join()
        } catch (_: InterruptedException) {}
        bgThread = null
        bgHandler = null
    }

    private fun chooseFrontCamera() {
        val ids = camMgr.cameraIdList
        var bestId: String? = null
        for (id in ids) {
            val chars = camMgr.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                bestId = id
                sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
                isFrontFacing = true
                break
            }
        }
        if (bestId == null) {
            // Fallback: pick any camera (shouldn’t happen on phones)
            bestId = ids.firstOrNull()
            val chars = bestId?.let { camMgr.getCameraCharacteristics(it) }
            sensorOrientation = chars?.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            isFrontFacing = (chars?.get(CameraCharacteristics.LENS_FACING)
                == CameraCharacteristics.LENS_FACING_FRONT)
        }
        cameraId = bestId
        Log.i(TAG, "Selected cameraId=$cameraId sensorOrientation=$sensorOrientation front=$isFrontFacing")
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        val id = cameraId ?: return
        camMgr.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                if (closed) { device.close(); return }
                camera = device
                createSession()
            }

            override fun onDisconnected(device: CameraDevice) {
                Log.w(TAG, "Camera disconnected")
                try { device.close() } catch (_: Throwable) {}
                camera = null
            }

            override fun onError(device: CameraDevice, error: Int) {
                Log.e(TAG, "Camera error: $error")
                try { device.close() } catch (_: Throwable) {}
                camera = null
            }
        }, bgHandler)
    }

    private fun createSession() {
        val cam = camera ?: return

        // Create YUV reader for ML
        val (w, h) = pickYuvSize(cam)
        imageReader = ImageReader.newInstance(w, h, ImageFormat.YUV_420_888, MAX_IMAGES).apply {
            setOnImageAvailableListener(readerListener, bgHandler)
        }

        val targets = listOf(imageReader!!.surface)
        try {
            cam.createCaptureSession(
                targets,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (closed) { safeClose(session); return }
                        captureSession = session
                        startRepeating()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Session configure failed")
                    }
                },
                bgHandler
            )
        } catch (t: Throwable) {
            Log.e(TAG, "createCaptureSession failed", t)
        }
    }

    private fun startRepeating() {
        val cam = camera ?: return
        val session = captureSession ?: return
        val request = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            imageReader?.surface?.let { addTarget(it) }
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        }.build()

        try {
            session.setRepeatingRequest(request, null, bgHandler)
        } catch (t: Throwable) {
            Log.e(TAG, "setRepeatingRequest failed", t)
        }
    }

    @WorkerThread
    private val readerListener = ImageReader.OnImageAvailableListener { reader ->
        val image = try { reader.acquireLatestImage() } catch (_: Throwable) { null } ?: return@OnImageAvailableListener
        if (closed) {
            try { image.close() } catch (_: Throwable) {}
            return@OnImageAvailableListener
        }
        val rotation = computeRotationDegrees(latestDisplayRotation, sensorOrientation, isFrontFacing)
        // hand the Image to the app; they must call markFrameDone(image)
        try {
            onFrame(image, rotation)
        } catch (t: Throwable) {
            Log.w(TAG, "onFrame callback threw", t)
            try { image.close() } catch (_: Throwable) {}
        }
    }

    private fun computeRotationDegrees(displayRotation: Int, sensorOrientation: Int, front: Boolean): Int {
        val uiDegrees = when (displayRotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        // ML Kit style compensation
        return if (front) {
            (sensorOrientation + uiDegrees) % 360
        } else {
            (sensorOrientation - uiDegrees + 360) % 360
        }
    }

    private fun pickYuvSize(device: CameraDevice): Pair<Int, Int> {
        // We requested targetSize; ensure both dims are > 0, fallback to a conservative 320x240
        val w = max(1, targetSize.width)
        val h = max(1, targetSize.height)
        return w to h
    }

    private fun safeClose(session: CameraCaptureSession) {
        try { session.close() } catch (_: Throwable) {}
    }
}

