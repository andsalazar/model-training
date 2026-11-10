# Retry starting a Runpod pod until the host has free GPU capacity.
#
# The persistent pod's host is often full even when the catalog reports stock in
# the data center, so a single start attempt fails with a 500. Retrying every
# minute for a bounded window is what worked previously (succeeded on the 7th
# attempt after about eight minutes).
param(
    [Parameter(Mandatory = $true)]
    [string]$PodId,
    [int]$MaxMinutes = 25,
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = "Stop"

$key = doppler secrets get RUNPOD_MODEL_TRAINING_API_KEY --project model-training --config dev --plain 2>$null
if (-not $key) {
    Write-Output "FAILED: could not read RUNPOD_MODEL_TRAINING_API_KEY from Doppler"
    exit 1
}

$headers = @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" }
$deadline = (Get-Date).AddMinutes($MaxMinutes)
$attempt = 0

while ((Get-Date) -lt $deadline) {
    $attempt++
    $stamp = (Get-Date).ToString("HH:mm:ss")

    try {
        $pod = Invoke-RestMethod -Method Get -Uri "https://rest.runpod.io/v1/pods/$PodId" -Headers $headers -TimeoutSec 30
        if ($pod.status -eq "RUNNING") {
            Write-Output "$stamp attempt ${attempt}: already RUNNING"
            Write-Output "SUCCESS"
            exit 0
        }
        if ($pod.status -in @("PROVISIONING", "STARTING")) {
            Write-Output "$stamp attempt ${attempt}: $($pod.status), waiting"
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $null = Invoke-RestMethod -Method Post -Uri "https://rest.runpod.io/v1/pods/$PodId/start" -Headers $headers -TimeoutSec 60
        Write-Output "$stamp attempt ${attempt}: start accepted"
        Write-Output "SUCCESS"
        exit 0
    }
    catch {
        $detail = $_.ErrorDetails.Message
        if (-not $detail) { $detail = $_.Exception.Message }
        Write-Output "$stamp attempt ${attempt}: $detail"
    }

    Start-Sleep -Seconds $IntervalSeconds
}

Write-Output "GAVE_UP after $attempt attempts in $MaxMinutes minutes"
exit 2
