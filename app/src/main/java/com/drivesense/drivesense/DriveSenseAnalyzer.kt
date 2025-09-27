package com.drivesense.drivesense

import android.graphics.PointF
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetector
import java.util.concurrent.Executor
import kotlin.math.abs

class DriveSenseAnalyzer(
    private val detector: FaceDetector,
    private val mainExecutor: Executor,
    private val onStateUpdated: (DriverState) -> Unit,
    private val closedEyesThresholdMs: Long = 1500L,
    private val minEyeOpenProbability: Float = 0.4f
) : ImageAnalysis.Analyzer {

    private var lastEyesClosedAt: Long = NO_TIMESTAMP
    private var lastState: DriverState = DriverState.Initializing
    private val leftEyeAspectFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val rightEyeAspectFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val leftEyeProbabilityFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)
    private val rightEyeProbabilityFilter = ExponentialMovingAverage(SMOOTHING_ALPHA)

    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        detector.process(inputImage)
            .addOnSuccessListener { faces ->
                val newState = evaluateState(faces)
                publishState(newState)
            }
            .addOnFailureListener { error ->
                publishState(DriverState.Error(error.localizedMessage ?: "Face detector error"))
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun evaluateState(faces: List<Face>): DriverState {
        if (faces.isEmpty()) {
            lastEyesClosedAt = NO_TIMESTAMP
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

        if (!classificationAvailable && !contourAvailable) {
            lastEyesClosedAt = NO_TIMESTAMP
            return DriverState.Attentive
        }

        val votesForClosed = sequenceOf(eyesClosedByProbability, eyesClosedByContour).count { it }
        val votesForOpen = sequenceOf(eyesOpenByProbability, eyesOpenByContour).count { it }

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
