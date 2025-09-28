package com.drivesense.drivesense.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.graphics.withSave
import java.util.concurrent.CopyOnWriteArrayList

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#4CAF50")
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 42f
        style = Paint.Style.FILL
    }

    private val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#66000000")
        style = Paint.Style.FILL
    }

    private val detections = CopyOnWriteArrayList<RoadObjectDetection>()

    fun updateDetections(newDetections: List<RoadObjectDetection>) {
        detections.clear()
        detections.addAll(newDetections)
        postInvalidateOnAnimation()
    }

    fun clearDetections() {
        detections.clear()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val width = width.toFloat()
        val height = height.toFloat()

        canvas.withSave {
            detections.forEach { detection ->
                val rect = RectF(
                    detection.bounds.left * width,
                    detection.bounds.top * height,
                    detection.bounds.right * width,
                    detection.bounds.bottom * height
                )

                canvas.drawRect(rect, boxPaint)

                val label = detection.label
                if (label.isNotEmpty()) {
                    val textPadding = 8f
                    val textWidth = textPaint.measureText(label)
                    val textHeight = textPaint.textSize
                    val backgroundTop = (rect.top - textHeight - textPadding * 2).coerceAtLeast(0f)
                    val backgroundRect = RectF(
                        rect.left,
                        backgroundTop,
                        rect.left + textWidth + textPadding * 2,
                        backgroundTop + textHeight + textPadding * 2
                    )
                    canvas.drawRoundRect(backgroundRect, 12f, 12f, backgroundPaint)
                    canvas.drawText(
                        label,
                        backgroundRect.left + textPadding,
                        backgroundRect.bottom - textPadding,
                        textPaint
                    )
                }
            }
        }
    }

    data class RoadObjectDetection(
        val bounds: RectF,
        val label: String
    )
}
