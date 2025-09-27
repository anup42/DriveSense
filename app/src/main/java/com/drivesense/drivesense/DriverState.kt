package com.drivesense.drivesense

sealed class DriverState {
    object Initializing : DriverState()
    object NoFace : DriverState()
    object Attentive : DriverState()
    data class Drowsy(val closedDurationMs: Long) : DriverState()
    data class Error(val reason: String) : DriverState()
}
