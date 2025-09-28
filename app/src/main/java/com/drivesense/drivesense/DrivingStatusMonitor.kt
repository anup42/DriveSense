package com.drivesense.drivesense

import android.Manifest
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import androidx.core.content.ContextCompat
import com.google.android.gms.location.ActivityRecognition
import com.google.android.gms.location.ActivityTransition
import com.google.android.gms.location.ActivityTransitionRequest
import com.google.android.gms.location.ActivityTransitionResult
import com.google.android.gms.location.DetectedActivity
import java.util.concurrent.Executor

class DrivingStatusMonitor(
    context: Context,
    private val callbackExecutor: Executor,
    private val onStatusChanged: (DrivingDetectionStatus) -> Unit
) {

    private val appContext = context.applicationContext
    private val activityRecognitionClient = ActivityRecognition.getClient(appContext)
    private val transitionRequest = ActivityTransitionRequest(
        listOf(
            ActivityTransition.Builder()
                .setActivityType(DetectedActivity.IN_VEHICLE)
                .setActivityTransition(ActivityTransition.ACTIVITY_TRANSITION_ENTER)
                .build(),
            ActivityTransition.Builder()
                .setActivityType(DetectedActivity.IN_VEHICLE)
                .setActivityTransition(ActivityTransition.ACTIVITY_TRANSITION_EXIT)
                .build()
        )
    )
    private val transitionAction = "${appContext.packageName}.DRIVING_TRANSITIONS"
    private val transitionIntent = Intent(transitionAction).setPackage(appContext.packageName)
    private val pendingIntentFlags = PendingIntent.FLAG_UPDATE_CURRENT or
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0
    private val transitionPendingIntent = PendingIntent.getBroadcast(
        appContext,
        REQUEST_CODE_TRANSITIONS,
        transitionIntent,
        pendingIntentFlags
    )

    private var receiverRegistered = false
    private var updatesRequested = false

    var status: DrivingDetectionStatus = DrivingDetectionStatus.UNKNOWN
        private set

    private val transitionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent == null || !ActivityTransitionResult.hasResult(intent)) {
                return
            }
            val result = ActivityTransitionResult.extractResult(intent) ?: return
            for (event in result.transitionEvents) {
                if (event.activityType != DetectedActivity.IN_VEHICLE) continue
                when (event.transitionType) {
                    ActivityTransition.ACTIVITY_TRANSITION_ENTER -> {
                        notifyStatus(DrivingDetectionStatus.DRIVING)
                    }

                    ActivityTransition.ACTIVITY_TRANSITION_EXIT -> {
                        notifyStatus(DrivingDetectionStatus.STATIONARY)
                    }
                }
            }
        }
    }

    fun start() {
        if (!hasActivityRecognitionPermission(appContext)) {
            notifyStatus(DrivingDetectionStatus.UNAVAILABLE)
            dispatchCurrentStatus()
            return
        }

        if (!receiverRegistered) {
            registerReceiver()
        }

        if (!updatesRequested) {
            activityRecognitionClient.requestActivityTransitionUpdates(transitionRequest, transitionPendingIntent)
                .addOnSuccessListener {
                    updatesRequested = true
                }
                .addOnFailureListener {
                    updatesRequested = false
                    notifyStatus(DrivingDetectionStatus.UNAVAILABLE)
                }
        }

        if (status != DrivingDetectionStatus.UNKNOWN) {
            notifyStatus(DrivingDetectionStatus.UNKNOWN)
        } else {
            dispatchCurrentStatus()
        }
    }

    fun stop() {
        if (updatesRequested) {
            activityRecognitionClient.removeActivityTransitionUpdates(transitionPendingIntent)
            updatesRequested = false
        }

        if (receiverRegistered) {
            appContext.unregisterReceiver(transitionReceiver)
            receiverRegistered = false
        }

        if (status != DrivingDetectionStatus.UNAVAILABLE) {
            notifyStatus(DrivingDetectionStatus.UNKNOWN)
        }
    }

    fun handlePermissionDenied() {
        notifyStatus(DrivingDetectionStatus.UNAVAILABLE)
    }

    private fun registerReceiver() {
        val filter = IntentFilter(transitionAction)
        ContextCompat.registerReceiver(
            appContext,
            transitionReceiver,
            filter,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        receiverRegistered = true
    }

    private fun notifyStatus(newStatus: DrivingDetectionStatus) {
        if (newStatus == status) return
        status = newStatus
        dispatchCurrentStatus()
    }

    private fun dispatchCurrentStatus() {
        callbackExecutor.execute { onStatusChanged(status) }
    }

    companion object {
        private const val REQUEST_CODE_TRANSITIONS = 1001

        fun hasActivityRecognitionPermission(context: Context): Boolean {
            return if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                true
            } else {
                ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
            }
        }
    }
}

enum class DrivingDetectionStatus {
    UNKNOWN,
    STATIONARY,
    DRIVING,
    UNAVAILABLE
}
