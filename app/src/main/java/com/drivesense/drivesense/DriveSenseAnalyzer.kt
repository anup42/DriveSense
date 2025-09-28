package com.drivesense.drivesense

import android.graphics.Bitmap
import android.content.Context
import android.graphics.PointF
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
import java.util.concurrent.Executor
import kotlin.LazyThreadSafetyMode
import kotlin.math.abs
import kotlin.math.hypot

class DriveSenseAnalyzer(
    private val context: Context,
    private val detector: FaceDetector,
    private val mainExecutor: Executor,
    private val onStateUpdated: (DriverState) -> Unit,
    private val onFrameProcessed: (() -> Unit)? = null,
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
            .setNumFaces(1)
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

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
        val mediaPipeMetrics = runCatching {
            imageProxy.toBitmap()?.let { detectWithMediaPipe(it) }
        }.getOrNull()
        detector.process(inputImage)
            .addOnSuccessListener { faces ->
                val newState = evaluateState(faces, mediaPipeMetrics)
                publishState(newState)
            }
            .addOnFailureListener { error ->
                publishState(DriverState.Error(error.localizedMessage ?: "Face detector error"))
            }
            .addOnCompleteListener {
                onFrameProcessed?.invoke()
                imageProxy.close()
            }
    }

    fun close() {
        if (faceLandmarkerInitialized) {
            faceLandmarker.close()
        }
    }

    private fun evaluateState(
        faces: List<Face>,
        mediaPipeMetrics: MediaPipeEyeMetrics?
    ): DriverState {
        if (faces.isEmpty()) {
            lastEyesClosedAt = NO_TIMESTAMP
            resetMediaPipeFilters()
            return DriverState.NoFace
        }

        val primaryFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: faces.first()
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
            return DriverState.Attentive
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
                DriverState.Drowsy(closedDuration)
            } else {
                DriverState.Attentive
            }
        }

        lastEyesClosedAt = NO_TIMESTAMP
        return DriverState.Attentive
    }

    private fun detectWithMediaPipe(bitmap: Bitmap): MediaPipeEyeMetrics? {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = faceLandmarker.detectForVideo(mpImage, SystemClock.uptimeMillis())
        if (result.faceLandmarks().isEmpty()) return null

        val landmarks = result.faceLandmarks()[0]
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

