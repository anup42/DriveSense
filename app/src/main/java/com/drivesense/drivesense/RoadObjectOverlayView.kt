package com.drivesense.drivesense

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import java.util.Locale
import kotlin.math.min

class RoadObjectOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val outlinePaintByCategory: Map<RoadObjectCategory, Paint>
    private val labelBackgroundPaintByCategory: Map<RoadObjectCategory, Paint>
    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 14f, resources.displayMetrics)
    }

    private val boxCornerRadius = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 6f, resources.displayMetrics)
    private val boxStrokeWidth = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 2f, resources.displayMetrics)
    private val labelPadding = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 4f, resources.displayMetrics)

    private var detectionResult: RoadObjectDetectionResult? = null
    private val rect = RectF()

    init {
        val vehicleColor = Color.parseColor("#FF9800")
        val pedestrianColor = Color.parseColor("#66BB6A")
        val animalColor = Color.parseColor("#EF5350")

        outlinePaintByCategory = mapOf(
            RoadObjectCategory.VEHICLE to createOutlinePaint(vehicleColor),
            RoadObjectCategory.PEDESTRIAN to createOutlinePaint(pedestrianColor),
            RoadObjectCategory.ANIMAL to createOutlinePaint(animalColor)
        )

        labelBackgroundPaintByCategory = mapOf(
            RoadObjectCategory.VEHICLE to createLabelBackgroundPaint(vehicleColor),
            RoadObjectCategory.PEDESTRIAN to createLabelBackgroundPaint(pedestrianColor),
            RoadObjectCategory.ANIMAL to createLabelBackgroundPaint(animalColor)
        )
    }

    fun render(result: RoadObjectDetectionResult?) {
        detectionResult = result
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val result = detectionResult ?: return
        val detections = result.detections
        if (detections.isEmpty()) return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        if (viewWidth <= 0f || viewHeight <= 0f) return

        val imageWidth = result.imageWidth.toFloat()
        val imageHeight = result.imageHeight.toFloat()
        if (imageWidth <= 0f || imageHeight <= 0f) return

        val scale = min(viewWidth / imageWidth, viewHeight / imageHeight)
        val scaledWidth = imageWidth * scale
        val scaledHeight = imageHeight * scale
        val offsetX = (viewWidth - scaledWidth) / 2f
        val offsetY = (viewHeight - scaledHeight) / 2f

        for (detection in detections) {
            val outlinePaint = outlinePaintByCategory[detection.category] ?: continue
            val labelBackgroundPaint = labelBackgroundPaintByCategory[detection.category] ?: continue

            val left = offsetX + detection.boundingBox.left * scaledWidth
            val top = offsetY + detection.boundingBox.top * scaledHeight
            val right = offsetX + detection.boundingBox.right * scaledWidth
            val bottom = offsetY + detection.boundingBox.bottom * scaledHeight

            rect.set(left, top, right, bottom)
            canvas.drawRoundRect(rect, boxCornerRadius, boxCornerRadius, outlinePaint)

            val label = buildLabel(detection)
            val textWidth = labelTextPaint.measureText(label)
            val textHeight = labelTextPaint.textSize

            var backgroundTop = top - textHeight - labelPadding * 2
            if (backgroundTop < offsetY) {
                backgroundTop = top
            }
            var backgroundLeft = left
            var backgroundRight = backgroundLeft + textWidth + labelPadding * 2
            if (backgroundRight > viewWidth - offsetX) {
                backgroundRight = viewWidth - offsetX
                backgroundLeft = backgroundRight - textWidth - labelPadding * 2
            }
            val backgroundBottom = backgroundTop + textHeight + labelPadding * 2

            rect.set(backgroundLeft, backgroundTop, backgroundRight, backgroundBottom)
            canvas.drawRoundRect(rect, boxCornerRadius, boxCornerRadius, labelBackgroundPaint)
            canvas.drawText(label, rect.left + labelPadding, rect.bottom - labelPadding, labelTextPaint)
        }
    }

    private fun buildLabel(detection: RoadObjectDetection): String {
        val confidencePercent = (detection.score * 100f).coerceIn(0f, 100f)
        val confidenceText = String.format(Locale.US, "%.0f%%", confidencePercent)
        return "${detection.label} $confidenceText"
    }

    private fun createOutlinePaint(color: Int): Paint {
        return Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = boxStrokeWidth
            this.color = color
        }
    }

    private fun createLabelBackgroundPaint(color: Int): Paint {
        val backgroundColor = Color.argb(180, Color.red(color), Color.green(color), Color.blue(color))
        return Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = backgroundColor
        }
    }
}
