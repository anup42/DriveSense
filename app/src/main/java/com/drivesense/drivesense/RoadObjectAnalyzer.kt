package com.drivesense.drivesense

import android.content.Context
import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.components.containers.Detection
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit

class RoadObjectAnalyzer(
    private val context: Context,
    private val callbackExecutor: Executor,
    private val onDetectionsUpdated: (RoadObjectDetectionResult?) -> Unit,
    private val onFrameProcessed: () -> Unit,
    private val scoreThreshold: Float = 0.5f,
    private val maxResults: Int = 5
) : ImageAnalysis.Analyzer {

    @Volatile
    var detectionEnabled: Boolean = false

    private var objectDetector: ObjectDetector? = null
    private var lastDeliveredNull = false

    override fun analyze(imageProxy: ImageProxy) {
        try {
            if (!detectionEnabled) {
                deliverNullOnce()
                return
            }

            val bitmap = imageProxy.toBitmap()
            if (bitmap == null) {
                deliverNullOnce()
                return
            }

            val imageWidth = bitmap.width
            val imageHeight = bitmap.height
            val mpImage = BitmapImageBuilder(bitmap).build()
            val timestampMs = TimeUnit.NANOSECONDS.toMillis(imageProxy.imageInfo.timestamp)
            val detectionResult = runCatching {
                ensureDetector().detectForVideo(mpImage, timestampMs)
            }.getOrNull()
            bitmap.recycle()

            if (detectionResult == null) {
                deliverNullOnce()
                return
            }

            val mappedDetections = detectionResult.detections().mapNotNull { detection ->
                mapDetection(detection, imageWidth, imageHeight)
            }

            val result = RoadObjectDetectionResult(mappedDetections, imageWidth, imageHeight)
            deliver(result)
            lastDeliveredNull = false
        } finally {
            onFrameProcessed()
            imageProxy.close()
        }
    }

    fun close() {
        objectDetector?.close()
        objectDetector = null
    }

    private fun mapDetection(detection: Detection, imageWidth: Int, imageHeight: Int): RoadObjectDetection? {
        val category = detection.categories().maxByOrNull { it.score() } ?: return null
        val mappedCategory = mapCategory(category.categoryName()) ?: return null

        val boundingBox = detection.boundingBox()
        val normalized = RectF(
            (boundingBox.left / imageWidth.toFloat()).coerceIn(0f, 1f),
            (boundingBox.top / imageHeight.toFloat()).coerceIn(0f, 1f),
            (boundingBox.right / imageWidth.toFloat()).coerceIn(0f, 1f),
            (boundingBox.bottom / imageHeight.toFloat()).coerceIn(0f, 1f)
        )
        if (normalized.width() <= 0f || normalized.height() <= 0f) return null

        val label = category.categoryName()
        return RoadObjectDetection(
            category = mappedCategory,
            label = label.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() },
            score = category.score(),
            boundingBox = normalized
        )
    }

    private fun ensureDetector(): ObjectDetector {
        val existing = objectDetector
        if (existing != null) return existing

        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(OBJECT_DETECTOR_ASSET)
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO)
            .setScoreThreshold(scoreThreshold)
            .setMaxResults(maxResults)
            .build()

        val detector = ObjectDetector.createFromOptions(context, options)
        objectDetector = detector
        return detector
    }

    private fun mapCategory(label: String): RoadObjectCategory? {
        return when (label.lowercase(Locale.US)) {
            "person" -> RoadObjectCategory.PEDESTRIAN
            "bicycle", "car", "motorcycle", "bus", "truck", "train" -> RoadObjectCategory.VEHICLE
            "dog", "cat", "bird", "cow", "horse", "sheep", "bear", "zebra", "giraffe", "elephant" -> RoadObjectCategory.ANIMAL
            else -> null
        }
    }

    private fun deliver(result: RoadObjectDetectionResult?) {
        callbackExecutor.execute { onDetectionsUpdated(result) }
    }

    private fun deliverNullOnce() {
        if (!lastDeliveredNull) {
            deliver(null)
            lastDeliveredNull = true
        }
    }

    companion object {
        private const val OBJECT_DETECTOR_ASSET = "efficientdet_lite0.tflite"
    }
}
