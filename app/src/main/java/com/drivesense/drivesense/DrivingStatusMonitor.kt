package com.drivesense.drivesense

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import java.util.concurrent.Executor
import kotlin.math.sqrt

class DrivingStatusMonitor(
    context: Context,
    private val callbackExecutor: Executor,
    private val onStatusChanged: (DrivingDetectionStatus) -> Unit,
    private val motionThreshold: Float = DEFAULT_MOTION_THRESHOLD,
    private val stationaryTimeoutMs: Long = DEFAULT_STATIONARY_TIMEOUT_MS,
    private val drivingActivationMs: Long = DEFAULT_DRIVING_ACTIVATION_MS
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val linearAccelerationSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    private var lastMotionTimestamp: Long = NO_TIMESTAMP
    private var drivingCandidateStart: Long = NO_TIMESTAMP

    var status: DrivingDetectionStatus =
        if (linearAccelerationSensor == null) DrivingDetectionStatus.UNAVAILABLE else DrivingDetectionStatus.UNKNOWN
        private set

    fun start() {
        linearAccelerationSensor ?: run {
            notifyStatus(DrivingDetectionStatus.UNAVAILABLE)
            callbackExecutor.execute { onStatusChanged(status) }
            return
        }

        lastMotionTimestamp = NO_TIMESTAMP
        drivingCandidateStart = NO_TIMESTAMP
        sensorManager.registerListener(
            this,
            linearAccelerationSensor,
            SensorManager.SENSOR_DELAY_NORMAL
        )
        callbackExecutor.execute { onStatusChanged(status) }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        lastMotionTimestamp = NO_TIMESTAMP
        drivingCandidateStart = NO_TIMESTAMP
        if (status != DrivingDetectionStatus.UNAVAILABLE) {
            notifyStatus(DrivingDetectionStatus.UNKNOWN)
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val magnitude = sqrt(
            event.values[0] * event.values[0] +
                event.values[1] * event.values[1] +
                event.values[2] * event.values[2]
        )
        val now = SystemClock.elapsedRealtime()

        if (magnitude >= motionThreshold) {
            lastMotionTimestamp = now
            if (status != DrivingDetectionStatus.DRIVING) {
                if (drivingCandidateStart == NO_TIMESTAMP) {
                    drivingCandidateStart = now
                }
                if (now - drivingCandidateStart >= drivingActivationMs) {
                    notifyStatus(DrivingDetectionStatus.DRIVING)
                }
            } else {
                drivingCandidateStart = now
            }
            return
        }

        drivingCandidateStart = NO_TIMESTAMP

        if (lastMotionTimestamp == NO_TIMESTAMP) {
            lastMotionTimestamp = now
        }

        val elapsedSinceMotion = now - lastMotionTimestamp
        if (elapsedSinceMotion >= stationaryTimeoutMs) {
            notifyStatus(DrivingDetectionStatus.STATIONARY)
        } else if (status == DrivingDetectionStatus.UNKNOWN) {
            notifyStatus(DrivingDetectionStatus.UNKNOWN)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun notifyStatus(newStatus: DrivingDetectionStatus) {
        if (newStatus == status) return
        status = newStatus
        callbackExecutor.execute { onStatusChanged(newStatus) }
    }

    companion object {
        private const val DEFAULT_MOTION_THRESHOLD = 1.2f
        private const val DEFAULT_STATIONARY_TIMEOUT_MS = 15000L
        private const val DEFAULT_DRIVING_ACTIVATION_MS = 2000L
        private const val NO_TIMESTAMP = -1L
    }
}

enum class DrivingDetectionStatus {
    UNKNOWN,
    STATIONARY,
    DRIVING,
    UNAVAILABLE
}
