package com.drivesense.drivesense.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.graphics.withSave

class FaceSelectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFB300")
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val bounds = RectF()
    @Volatile
    private var hasBounds: Boolean = false

    fun updateBounds(newBounds: RectF?) {
        if (newBounds == null) {
            if (!hasBounds) {
                return
            }
            hasBounds = false
            postInvalidateOnAnimation()
            return
        }

        bounds.set(newBounds)
        hasBounds = true
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!hasBounds) {
            return
        }

        val width = width.toFloat()
        val height = height.toFloat()
        if (width <= 0f || height <= 0f) {
            return
        }

        canvas.withSave {
            val rect = RectF(
                bounds.left * width,
                bounds.top * height,
                bounds.right * width,
                bounds.bottom * height
            )
            canvas.drawRect(rect, boxPaint)
        }
    }
}
