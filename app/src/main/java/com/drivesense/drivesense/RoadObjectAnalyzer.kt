package com.drivesense.drivesense

import android.graphics.Rect
import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.drivesense.drivesense.ui.DetectionOverlayView.RoadObjectDetection
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetector
import java.util.Locale
import java.util.concurrent.Executor

class RoadObjectAnalyzer(
    private val detector: ObjectDetector,
    private val mainExecutor: Executor,
    private val onDetectionsUpdated: (List<RoadObjectDetection>) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val image = InputImage.fromMediaImage(mediaImage, rotationDegrees)
        val imageWidth = imageProxy.width
        val imageHeight = imageProxy.height

        detector
            .process(image)
            .addOnSuccessListener { detectedObjects ->
                val detections = detectedObjects.mapNotNull { detectedObject ->
                    val label = detectedObject.classificationLabel() ?: return@mapNotNull null
                    val bounds = detectedObject.boundingBox.toNormalizedBounds(
                        imageWidth,
                        imageHeight,
                        rotationDegrees
                    ) ?: return@mapNotNull null
                    RoadObjectDetection(bounds, label)
                }
                mainExecutor.execute { onDetectionsUpdated(detections) }
            }
            .addOnFailureListener {
                mainExecutor.execute { onDetectionsUpdated(emptyList()) }
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun DetectedObject.classificationLabel(): String? {
        val label = labels
            .filter { it.confidence >= MIN_CONFIDENCE }
            .maxByOrNull { it.confidence }
            ?: return null
        val normalizedLabel = label.text.lowercase(Locale.US)
        val match = when {
            normalizedLabel.contains("person") || normalizedLabel.contains("pedestrian") -> "Pedestrian"
            normalizedLabel.contains("cat") || normalizedLabel.contains("dog") || normalizedLabel.contains("animal") -> "Animal"
            normalizedLabel.contains("car") || normalizedLabel.contains("vehicle") ||
                normalizedLabel.contains("bus") || normalizedLabel.contains("truck") ||
                normalizedLabel.contains("bike") || normalizedLabel.contains("bicycle") ||
                normalizedLabel.contains("motorcycle") -> "Vehicle"
            else -> null
        }
        return match
    }

    private fun Rect.toNormalizedBounds(
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): RectF? {
        if (width() <= 0 || height() <= 0) {
            return null
        }
        val (normalizedLeft, normalizedTop, normalizedRight, normalizedBottom) = when (rotationDegrees) {
            0 -> listOf(
                left / imageWidth.toFloat(),
                top / imageHeight.toFloat(),
                right / imageWidth.toFloat(),
                bottom / imageHeight.toFloat()
            )
            90 -> listOf(
                top / imageHeight.toFloat(),
                (imageWidth - right) / imageWidth.toFloat(),
                bottom / imageHeight.toFloat(),
                (imageWidth - left) / imageWidth.toFloat()
            )
            180 -> listOf(
                (imageWidth - right) / imageWidth.toFloat(),
                (imageHeight - bottom) / imageHeight.toFloat(),
                (imageWidth - left) / imageWidth.toFloat(),
                (imageHeight - top) / imageHeight.toFloat()
            )
            270 -> listOf(
                (imageHeight - bottom) / imageHeight.toFloat(),
                left / imageWidth.toFloat(),
                (imageHeight - top) / imageHeight.toFloat(),
                right / imageWidth.toFloat()
            )
            else -> listOf(
                left / imageWidth.toFloat(),
                top / imageHeight.toFloat(),
                right / imageWidth.toFloat(),
                bottom / imageHeight.toFloat()
            )
        }

        val leftClamped = normalizedLeft.coerceIn(0f, 1f)
        val topClamped = normalizedTop.coerceIn(0f, 1f)
        val rightClamped = normalizedRight.coerceIn(0f, 1f)
        val bottomClamped = normalizedBottom.coerceIn(0f, 1f)

        return RectF(leftClamped, topClamped, rightClamped, bottomClamped)
    }

    companion object {
        private const val MIN_CONFIDENCE = 0.4f
    }
}
