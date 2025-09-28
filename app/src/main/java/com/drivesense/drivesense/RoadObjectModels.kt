package com.drivesense.drivesense

import android.graphics.RectF

enum class RoadObjectCategory {
    VEHICLE,
    PEDESTRIAN,
    ANIMAL
}

data class RoadObjectDetection(
    val category: RoadObjectCategory,
    val label: String,
    val score: Float,
    val boundingBox: RectF
)

data class RoadObjectDetectionResult(
    val detections: List<RoadObjectDetection>,
    val imageWidth: Int,
    val imageHeight: Int
)
