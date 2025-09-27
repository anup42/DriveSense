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

class DriveSenseAnalyzer(
    private val detector: FaceDetector,
    private val mainExecutor: Executor,
    private val onStateUpdated: (DriverState) -> Unit,
    private val closedEyesThresholdMs: Long = 1500L,
    private val minEyeOpenProbability: Float = 0.4f
) : ImageAnalysis.Analyzer {

    private var lastEyesClosedAt: Long = NO_TIMESTAMP
    private var lastState: DriverState = DriverState.Initializing

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
        val leftEyeProbability = primaryFace.leftEyeOpenProbability ?: UNKNOWN_PROBABILITY
        val rightEyeProbability = primaryFace.rightEyeOpenProbability ?: UNKNOWN_PROBABILITY

        val classificationAvailable = leftEyeProbability >= 0f && rightEyeProbability >= 0f
        val eyesClosedByProbability = classificationAvailable &&
            leftEyeProbability < minEyeOpenProbability &&
            rightEyeProbability < minEyeOpenProbability
        val eyesOpenByProbability = classificationAvailable &&
            leftEyeProbability >= OPEN_EYE_PROB_THRESHOLD &&
            rightEyeProbability >= OPEN_EYE_PROB_THRESHOLD

        val leftEyeAspectRatio = computeEyeAspectRatio(primaryFace.getContour(FaceContour.LEFT_EYE)?.points)
        val rightEyeAspectRatio = computeEyeAspectRatio(primaryFace.getContour(FaceContour.RIGHT_EYE)?.points)
        val contourAvailable = leftEyeAspectRatio != null && rightEyeAspectRatio != null
        val eyesClosedByContour = contourAvailable &&
            leftEyeAspectRatio < EYE_ASPECT_CLOSED_THRESHOLD &&
            rightEyeAspectRatio < EYE_ASPECT_CLOSED_THRESHOLD
        val eyesOpenByContour = contourAvailable &&
            (leftEyeAspectRatio >= EYE_ASPECT_OPEN_THRESHOLD || rightEyeAspectRatio >= EYE_ASPECT_OPEN_THRESHOLD)

        if (!classificationAvailable && !contourAvailable) {
            lastEyesClosedAt = NO_TIMESTAMP
            return DriverState.Attentive
        }

        val eyesClosed = when {
            eyesClosedByProbability && eyesOpenByContour -> false
            eyesClosedByProbability -> true
            eyesClosedByContour -> true
            eyesOpenByProbability -> false
            eyesOpenByContour -> false
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

        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE

        for (point in eyePoints) {
            if (point.x < minX) minX = point.x
            if (point.x > maxX) maxX = point.x
            if (point.y < minY) minY = point.y
            if (point.y > maxY) maxY = point.y
        }

        val horizontal = maxX - minX
        val vertical = maxY - minY
        if (horizontal <= 0f) return null
        return vertical / horizontal
    }

    private fun publishState(state: DriverState) {
        if (state == lastState) return
        lastState = state
        mainExecutor.execute { onStateUpdated(state) }
    }

    companion object {
        private const val UNKNOWN_PROBABILITY = -1f
        private const val NO_TIMESTAMP = -1L
        private const val MIN_CONTOUR_POINTS = 4
        private const val EYE_ASPECT_CLOSED_THRESHOLD = 0.18f
        private const val EYE_ASPECT_OPEN_THRESHOLD = 0.26f
        private const val OPEN_EYE_PROB_THRESHOLD = 0.55f
    }
}
