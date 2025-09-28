package com.drivesense.drivesense

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

fun ImageProxy.toBitmap(): Bitmap? {
    val planes = planes
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
    val width = width
    val height = height
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
    return bitmap.rotate(imageInfo.rotationDegrees)
}

private fun Bitmap.rotate(degrees: Int): Bitmap {
    if (degrees == 0) return this
    val matrix = Matrix()
    matrix.postRotate(degrees.toFloat())
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}
