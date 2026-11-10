# Watch for GPU capacity and start the persistent pod as soon as it frees.
#
# The EUR-IS-1 host holding the pod has been full for a long stretch. Rather than
# poll by hand, this loop retries the start endpoint and reports each attempt, so
# a long wait leaves a readable trail instead of a silent hang.
#
# Usage:
#   .\scripts\watch_pod_capacity.ps1 -PodId <pod-id> -MaxMinutes 120

param(
    [int]$MaxMinutes = 120,
    [Parameter(Mandatory = $true)]
    [string]$PodId,
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = "Stop"

$key = (doppler secrets get RUNPOD_MODEL_TRAINING_API_KEY `
        --project model-training --config dev --plain 2>&1 | Out-String).Trim()
if (-not $key) {
    Write-Output "DOPPLER_FAILED"
    exit 1
}

$headers = @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" }
$deadline = (Get-Date).AddMinutes($MaxMinutes)
$attempt = 0

while ((Get-Date) -lt $deadline) {
    $attempt++
    $stamp = (Get-Date).ToString("HH:mm:ss")

    try {
        Invoke-RestMethod -Method Post `
            -Uri "https://rest.runpod.io/v1/pods/$PodId/start" `
            -Headers $headers -TimeoutSec 60 -ErrorAction Stop | Out-Null
        Write-Output "[$attempt] $stamp START_ACCEPTED"
    }
    catch {
        Write-Output "[$attempt] $stamp full"
    }

    Start-Sleep -Seconds 15

    try {
        $pod = Invoke-RestMethod -Method Get `
            -Uri "https://rest.runpod.io/v1/pods/$PodId" `
            -Headers $headers -TimeoutSec 30
        if ($pod.desiredStatus -eq "RUNNING") {
            Write-Output "RUNNING after $attempt attempt(s)"
            Write-Output "pod=$PodId"
            exit 0
        }
    }
    catch {
        Write-Output "[$attempt] $stamp status check failed"
    }

    Start-Sleep -Seconds ($IntervalSeconds - 15)
}

Write-Output "TIMEOUT after $attempt attempt(s); host still full"
exit 2
