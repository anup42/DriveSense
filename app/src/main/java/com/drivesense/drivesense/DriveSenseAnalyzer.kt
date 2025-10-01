package com.drivesense.drivesense

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.media.Image
import android.graphics.YuvImage
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetector
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executor
import kotlin.LazyThreadSafetyMode
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

class DriveSenseAnalyzer(
    private val context: Context,
    private val detector: FaceDetector,
    private val mainExecutor: Executor,
    private val onStateUpdated: (DriverState) -> Unit,
    private val onDriverFaceUpdated: (RectF?) -> Unit = {},
    private val mirrorPreview: Boolean = false,
    private val closedEyesThresholdMs: Long = 1500L,
    private val minEyeOpenProbability: Float = 0.4f
) : ImageAnalysis.Analyzer {

    private var lastEyesClosedAt: Long = NO_TIMESTAMP
    private var lastState: DriverState = DriverState.Initializing
    private val leftEyeAspectFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val rightEyeAspectFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val leftEyeProbabilityFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val rightEyeProbabilityFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val mediaPipeLeftEarFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val mediaPipeRightEarFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val mediaPipeLeftProbabilityFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val mediaPipeRightProbabilityFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)

    private var faceLandmarkerInitialized = false
    private val faceLandmarker: FaceLandmarker by lazy(LazyThreadSafetyMode.NONE) {
        faceLandmarkerInitialized = true
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(FACE_LANDMARKER_ASSET)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumFaces(MAX_MEDIAPIPE_FACES)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .build()
        FaceLandmarker.createFromOptions(context, options)
    }

    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        processFrame(
            mediaImage = mediaImage,
            rotationDegrees = imageProxy.imageInfo.rotationDegrees,
            bitmapProvider = { imageProxyToBitmap(imageProxy) },
            onClose = { imageProxy.close() }
        )
    }

    fun analyze(mediaImage: Image, rotationDegrees: Int, onClose: () -> Unit) {
        processFrame(
            mediaImage = mediaImage,
            rotationDegrees = rotationDegrees,
            bitmapProvider = { mediaImageToBitmap(mediaImage, rotationDegrees) },
            onClose = onClose
        )
    }

    private fun processFrame(
        mediaImage: Image,
        rotationDegrees: Int,
        bitmapProvider: () -> Bitmap?,
        onClose: () -> Unit
    ) {
        val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
        val frameInfo = FrameInfo(
            width = mediaImage.width,
            height = mediaImage.height,
            rotationDegrees = rotationDegrees
        )
        val mediaPipeMetrics = runCatching {
            bitmapProvider()?.let { detectWithMediaPipe(it) }
        }.getOrNull()
        detector.process(inputImage)
            .addOnSuccessListener { faces ->
                val evaluation = evaluateState(faces, mediaPipeMetrics, frameInfo)
                publishDriverFace(evaluation.driverFaceBounds)
                publishState(evaluation.state)
            }
            .addOnFailureListener { error ->
                publishDriverFace(null)
                publishState(DriverState.Error(error.localizedMessage ?: "Face detector error"))
            }
            .addOnCompleteListener {
                onClose()
            }
    }

    fun close() {
        publishDriverFace(null)
        if (faceLandmarkerInitialized) {
            faceLandmarker.close()
        }
    }


    private fun evaluateState(
        faces: List<Face>,
        mediaPipeMetrics: MediaPipeEyeMetrics?,
        frameInfo: FrameInfo,
    ): EvaluationResult {
        if (faces.isEmpty()) {
            lastEyesClosedAt = NO_TIMESTAMP
            resetMediaPipeFilters()
            return EvaluationResult(DriverState.NoFace, null)
        }

        val primaryFace = faces
            .maxByOrNull { it.boundingBox.width().toLong() * it.boundingBox.height().toLong() }
            ?: faces.first()
        val normalizedBounds = primaryFace.boundingBox.toNormalizedBounds(frameInfo, mirrorPreview)

        val leftEyeProbability = primaryFace.leftEyeOpenProbability
        val rightEyeProbability = primaryFace.rightEyeOpenProbability
        val smoothedLeftEyeProbability = leftEyeProbability?.takeIf { it >= 0f }?.let {
            leftEyeProbabilityFilter.update(it)
        } ?: run {
            leftEyeProbabilityFilter.reset()
            null
        }
        val smoothedRightEyeProbability = rightEyeProbability?.takeIf { it >= 0f }?.let {
            rightEyeProbabilityFilter.update(it)
        } ?: run {
            rightEyeProbabilityFilter.reset()
            null
        }

        val classificationAvailable = smoothedLeftEyeProbability != null && smoothedRightEyeProbability != null
        val eyesClosedByProbability = if (classificationAvailable) {
            smoothedLeftEyeProbability!! < minEyeOpenProbability &&
                smoothedRightEyeProbability!! < minEyeOpenProbability
        } else {
            false
        }
        val eyesOpenByProbability = if (classificationAvailable) {
            smoothedLeftEyeProbability!! >= OPEN_EYE_PROB_THRESHOLD &&
                smoothedRightEyeProbability!! >= OPEN_EYE_PROB_THRESHOLD
        } else {
            false
        }

        val leftEyeAspectRatio = computeEyeAspectRatio(primaryFace.getContour(FaceContour.LEFT_EYE)?.points)
        val rightEyeAspectRatio = computeEyeAspectRatio(primaryFace.getContour(FaceContour.RIGHT_EYE)?.points)
        val smoothedLeftAspectRatio = leftEyeAspectRatio?.let { leftEyeAspectFilter.update(it) } ?: run {
            leftEyeAspectFilter.reset()
            null
        }
        val smoothedRightAspectRatio = rightEyeAspectRatio?.let { rightEyeAspectFilter.update(it) } ?: run {
            rightEyeAspectFilter.reset()
            null
        }

        val contourAvailable = smoothedLeftAspectRatio != null && smoothedRightAspectRatio != null
        val eyesClosedByContour = smoothedLeftAspectRatio?.let { leftRatio ->
            smoothedRightAspectRatio?.let { rightRatio ->
                leftRatio < EYE_ASPECT_CLOSED_THRESHOLD && rightRatio < EYE_ASPECT_CLOSED_THRESHOLD
            }
        } ?: false
        val eyesOpenByContour = listOfNotNull(smoothedLeftAspectRatio, smoothedRightAspectRatio)
            .any { it >= EYE_ASPECT_OPEN_THRESHOLD }

        val mediaPipeLeftEar = mediaPipeMetrics?.leftEar?.let { mediaPipeLeftEarFilter.update(it) } ?: run {
            mediaPipeLeftEarFilter.reset()
            null
        }
        val mediaPipeRightEar = mediaPipeMetrics?.rightEar?.let { mediaPipeRightEarFilter.update(it) } ?: run {
            mediaPipeRightEarFilter.reset()
            null
        }
        val mediaPipeLeftProbability = mediaPipeMetrics?.leftOpenProbability?.let {
            mediaPipeLeftProbabilityFilter.update(it)
        } ?: run {
            mediaPipeLeftProbabilityFilter.reset()
            null
        }
        val mediaPipeRightProbability = mediaPipeMetrics?.rightOpenProbability?.let {
            mediaPipeRightProbabilityFilter.update(it)
        } ?: run {
            mediaPipeRightProbabilityFilter.reset()
            null
        }

        val mediaPipeAspectAvailable = mediaPipeLeftEar != null && mediaPipeRightEar != null
        val mediaPipeProbabilityAvailable = mediaPipeLeftProbability != null && mediaPipeRightProbability != null

        val mediaPipeEyesClosedByAspect = if (mediaPipeAspectAvailable) {
            mediaPipeLeftEar!! < MEDIAPIPE_EYE_ASPECT_CLOSED_THRESHOLD &&
                mediaPipeRightEar!! < MEDIAPIPE_EYE_ASPECT_CLOSED_THRESHOLD
        } else {
            false
        }
        val mediaPipeEyesOpenByAspect = listOfNotNull(mediaPipeLeftEar, mediaPipeRightEar)
            .any { it >= MEDIAPIPE_EYE_ASPECT_OPEN_THRESHOLD }

        val mediaPipeEyesClosedByProbability = if (mediaPipeProbabilityAvailable) {
            mediaPipeLeftProbability!! < MEDIAPIPE_MIN_EYE_OPEN_PROBABILITY &&
                mediaPipeRightProbability!! < MEDIAPIPE_MIN_EYE_OPEN_PROBABILITY
        } else {
            false
        }
        val mediaPipeEyesOpenByProbability = listOfNotNull(mediaPipeLeftProbability, mediaPipeRightProbability)
            .any { it >= MEDIAPIPE_OPEN_EYE_PROB_THRESHOLD }

        if (!classificationAvailable && !contourAvailable &&
            !mediaPipeAspectAvailable && !mediaPipeProbabilityAvailable
        ) {
            lastEyesClosedAt = NO_TIMESTAMP
            return EvaluationResult(DriverState.Attentive, normalizedBounds)
        }

        val votesForClosed = listOf(
            eyesClosedByProbability,
            eyesClosedByContour,
            mediaPipeEyesClosedByAspect,
            mediaPipeEyesClosedByProbability
        ).count { it }
        val votesForOpen = listOf(
            eyesOpenByProbability,
            eyesOpenByContour,
            mediaPipeEyesOpenByAspect,
            mediaPipeEyesOpenByProbability
        ).count { it }

        val eyesClosed = when {
            votesForClosed == 0 -> false
            votesForOpen > votesForClosed -> false
            votesForClosed > votesForOpen -> true
            else -> false
        }

        if (eyesClosed) {
            if (lastEyesClosedAt == NO_TIMESTAMP) {
                lastEyesClosedAt = SystemClock.elapsedRealtime()
            }
            val closedDuration = SystemClock.elapsedRealtime() - lastEyesClosedAt
            return if (closedDuration >= closedEyesThresholdMs) {
                EvaluationResult(DriverState.Drowsy(closedDuration), normalizedBounds)
            } else {
                EvaluationResult(DriverState.Attentive, normalizedBounds)
            }
        }

        lastEyesClosedAt = NO_TIMESTAMP
        return EvaluationResult(DriverState.Attentive, normalizedBounds)
    }

    private fun detectWithMediaPipe(bitmap: Bitmap): MediaPipeEyeMetrics? {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = faceLandmarker.detectForVideo(mpImage, SystemClock.uptimeMillis())
        val landmarksList = result.faceLandmarks()
        if (landmarksList.isEmpty()) return null

        val landmarks = landmarksList.maxByOrNull { computeNormalizedArea(it) } ?: return null
        val leftEar = ear(
            landmarks,
            pOuter = 33,
            pInner = 133,
            pUpA = 160,
            pDownA = 144,
            pUpB = 158,
            pDownB = 153
        )
        val rightEar = ear(
            landmarks,
            pOuter = 362,
            pInner = 263,
            pUpA = 385,
            pDownA = 380,
            pUpB = 387,
            pDownB = 373
        )

        val leftProbability = mapEarToOpenProb(leftEar)
        val rightProbability = mapEarToOpenProb(rightEar)

        return MediaPipeEyeMetrics(
            leftEar = leftEar,
            rightEar = rightEar,
            leftOpenProbability = leftProbability,
            rightOpenProbability = rightProbability
        )
    }

    private fun computeNormalizedArea(landmarks: List<NormalizedLandmark>): Float {
        if (landmarks.isEmpty()) return 0f
        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        landmarks.forEach { landmark ->
            val x = landmark.x()
            val y = landmark.y()
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }
        val width = (maxX - minX).coerceAtLeast(0f)
        val height = (maxY - minY).coerceAtLeast(0f)
        return width * height
    }

    private fun ear(
        landmarks: List<NormalizedLandmark>,
        pOuter: Int,
        pInner: Int,
        pUpA: Int,
        pDownA: Int,
        pUpB: Int,
        pDownB: Int
    ): Float {
        fun distance(i: Int, j: Int): Float {
            val a = landmarks[i]
            val b = landmarks[j]
            return hypot(a.x() - b.x(), a.y() - b.y())
        }

        val horizontal = distance(pOuter, pInner)
        if (horizontal <= 1e-6f) {
            return 0f
        }

        val verticalA = distance(pUpA, pDownA)
        val verticalB = distance(pUpB, pDownB)
        val ear = ((verticalA + verticalB) / (2f * horizontal)).coerceIn(0f, 1f)
        return ear
    }

    private fun mapEarToOpenProb(ear: Float): Float {
        val clamped = ear.coerceIn(0f, 1f)
        val t = ((clamped - MEDIAPIPE_EAR_CLOSED) / (MEDIAPIPE_EAR_OPEN - MEDIAPIPE_EAR_CLOSED))
            .coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val mediaImage = imageProxy.image ?: return null
        return mediaImageToBitmap(mediaImage, imageProxy.imageInfo.rotationDegrees)
    }

    private fun mediaImageToBitmap(image: Image, rotationDegrees: Int): Bitmap? {
        val planes = image.planes
        if (planes.size < 3) return null

        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)

        val pixelStride = planes[2].pixelStride
        val rowStride = planes[2].rowStride
        val width = image.width
        val height = image.height
        var offset = ySize

        if (pixelStride == 2 && rowStride == width) {
            vBuffer.get(nv21, offset, vSize)
            offset += vSize
            uBuffer.get(nv21, offset, uSize)
        } else {
            val vBytes = ByteArray(vSize)
            val uBytes = ByteArray(uSize)
            vBuffer.get(vBytes)
            uBuffer.get(uBytes)
            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    val vuIndex = row * rowStride + col * pixelStride
                    nv21[offset++] = vBytes[vuIndex]
                    nv21[offset++] = uBytes[vuIndex]
                }
            }
        }

        yBuffer.rewind()
        uBuffer.rewind()
        vBuffer.rewind()

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val jpegBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
        return bitmap.rotate(rotationDegrees)
    }

    private fun Bitmap.rotate(degrees: Int): Bitmap {
        if (degrees == 0) return this
        val matrix = Matrix()
        matrix.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    private fun resetMediaPipeFilters() {
        mediaPipeLeftEarFilter.reset()
        mediaPipeRightEarFilter.reset()
        mediaPipeLeftProbabilityFilter.reset()
        mediaPipeRightProbabilityFilter.reset()
    }

    private data class MediaPipeEyeMetrics(
        val leftEar: Float,
        val rightEar: Float,
        val leftOpenProbability: Float,
        val rightOpenProbability: Float
    )

    private data class EvaluationResult(
        val state: DriverState,
        val driverFaceBounds: RectF?
    )

    private data class FrameInfo(
        val width: Int,
        val height: Int,
        val rotationDegrees: Int
    )

    private fun publishDriverFace(bounds: RectF?) {
        val copy = bounds?.let { RectF(it) }
        mainExecutor.execute { onDriverFaceUpdated(copy) }
    }

    private fun Rect.toNormalizedBounds(
        frameInfo: FrameInfo,
        mirrorHorizontally: Boolean,
    ): RectF? {
        if (width() <= 0 || height() <= 0) {
            return null
        }
        val imageWidth = frameInfo.width.toFloat()
        val imageHeight = frameInfo.height.toFloat()
        if (imageWidth <= 0f || imageHeight <= 0f) {
            return null
        }

        val rotation = ((frameInfo.rotationDegrees % 360) + 360) % 360
        val (normalizedLeft, normalizedTop, normalizedRight, normalizedBottom) = when (rotation) {
            0 -> listOf(
                left / imageWidth,
                top / imageHeight,
                right / imageWidth,
                bottom / imageHeight
            )
            90 -> listOf(
                top / imageHeight,
                (imageWidth - right) / imageWidth,
                bottom / imageHeight,
                (imageWidth - left) / imageWidth
            )
            180 -> listOf(
                (imageWidth - right) / imageWidth,
                (imageHeight - bottom) / imageHeight,
                (imageWidth - left) / imageWidth,
                (imageHeight - top) / imageHeight
            )
            270 -> listOf(
                (imageHeight - bottom) / imageHeight,
                left / imageWidth,
                (imageHeight - top) / imageHeight,
                right / imageWidth
            )
            else -> listOf(
                left / imageWidth,
                top / imageHeight,
                right / imageWidth,
                bottom / imageHeight
            )
        }

        var leftNorm = normalizedLeft.coerceIn(0f, 1f)
        var topNorm = normalizedTop.coerceIn(0f, 1f)
        var rightNorm = normalizedRight.coerceIn(0f, 1f)
        var bottomNorm = normalizedBottom.coerceIn(0f, 1f)

        if (mirrorHorizontally) {
            val mirroredLeft = 1f - rightNorm
            val mirroredRight = 1f - leftNorm
            leftNorm = mirroredLeft
            rightNorm = mirroredRight
        }

        val finalLeft = min(leftNorm, rightNorm).coerceIn(0f, 1f)
        val finalRight = max(leftNorm, rightNorm).coerceIn(0f, 1f)
        val finalTop = min(topNorm, bottomNorm).coerceIn(0f, 1f)
        val finalBottom = max(topNorm, bottomNorm).coerceIn(0f, 1f)

        if (finalLeft >= finalRight || finalTop >= finalBottom) {
            return null
        }

        return RectF(finalLeft, finalTop, finalRight, finalBottom)
    }

    private fun computeEyeAspectRatio(points: List<PointF>?): Float? {
        val eyePoints = points ?: return null
        if (eyePoints.size < MIN_CONTOUR_POINTS) return null

        val centerY = eyePoints.sumOf { it.y.toDouble() }.toFloat() / eyePoints.size
        val topPoints = eyePoints.filter { it.y <= centerY }
        val bottomPoints = eyePoints.filter { it.y > centerY }
        if (topPoints.isEmpty() || bottomPoints.isEmpty()) return null

        val leftmost = eyePoints.minByOrNull { it.x }?.x ?: return null
        val rightmost = eyePoints.maxByOrNull { it.x }?.x ?: return null
        val horizontal = rightmost - leftmost
        if (horizontal <= 0f) return null

        val verticalSamples = mutableListOf<Float>()
        for (top in topPoints) {
            val pairedBottom = bottomPoints.minByOrNull { abs(it.x - top.x) } ?: continue
            verticalSamples += abs(pairedBottom.y - top.y)
        }

        if (verticalSamples.size < MIN_VERTICAL_SAMPLES) return null

        val vertical = computeTrimmedMean(verticalSamples, VERTICAL_TRIM_RATIO)
        if (vertical <= 0f) return null

        return (vertical / horizontal).coerceAtLeast(0f)
    }

    private fun computeTrimmedMean(values: List<Float>, trimRatio: Float): Float {
        if (values.isEmpty()) return 0f
        if (values.size == 1) return values.first()

        val sorted = values.sorted()
        val trimCount = (sorted.size * trimRatio).toInt().coerceAtMost(sorted.lastIndex / 2)
        val fromIndex = trimCount
        val toIndex = sorted.size - trimCount
        if (toIndex <= fromIndex) {
            return sorted.average().toFloat()
        }
        val trimmed = sorted.subList(fromIndex, toIndex)
        return trimmed.average().toFloat()
    }

    private fun publishState(state: DriverState) {
        if (state == lastState) return
        lastState = state
        mainExecutor.execute { onStateUpdated(state) }
    }

    companion object {
        private const val NO_TIMESTAMP = -1L
        private const val MIN_CONTOUR_POINTS = 4
        private const val MIN_VERTICAL_SAMPLES = 3
        private const val EYE_ASPECT_CLOSED_THRESHOLD = 0.18f
        private const val EYE_ASPECT_OPEN_THRESHOLD = 0.26f
        private const val OPEN_EYE_PROB_THRESHOLD = 0.55f
        private const val VERTICAL_TRIM_RATIO = 0.2f
        private const val SMOOTHING_ALPHA = 0.35f
        private const val FACE_LANDMARKER_ASSET = "face_landmarker.task"
        private const val MAX_MEDIAPIPE_FACES = 3
        private const val MEDIAPIPE_EAR_CLOSED = 0.20f
        private const val MEDIAPIPE_EAR_OPEN = 0.30f
        private const val MEDIAPIPE_EYE_ASPECT_CLOSED_THRESHOLD = 0.20f
        private const val MEDIAPIPE_EYE_ASPECT_OPEN_THRESHOLD = 0.30f
        private const val MEDIAPIPE_MIN_EYE_OPEN_PROBABILITY = 0.30f
        private const val MEDIAPIPE_OPEN_EYE_PROB_THRESHOLD = 0.60f
    }
}

private class ExponentialMovingAverage(private val alpha: Float) {
    private var value: Float? = null

    fun update(newValue: Float): Float {
        val current = value
        val updated = if (current == null) {
            newValue
        } else {
            alpha * newValue + (1 - alpha) * current
        }
        value = updated
        return updated
    }

    fun reset() {
        value = null
    }
}
