package com.drivesense.drivesense

import android.os.SystemClock

class FpsTracker(private val smoothingFactor: Double = 0.2) {
    private var lastTimestampMs: Long = 0L
    private var smoothedFps: Double = 0.0

    fun tick(): Double {
        val now = SystemClock.elapsedRealtime()
        if (lastTimestampMs != 0L) {
            val deltaMs = now - lastTimestampMs
            if (deltaMs > 0) {
                val instantaneousFps = 1000.0 / deltaMs
                smoothedFps = if (smoothedFps == 0.0) {
                    instantaneousFps
                } else {
                    smoothedFps + (instantaneousFps - smoothedFps) * smoothingFactor
                }
            }
        }
        lastTimestampMs = now
        return smoothedFps
    }

    fun reset() {
        lastTimestampMs = 0L
        smoothedFps = 0.0
    }
}
