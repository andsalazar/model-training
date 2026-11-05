# Watch two pods for capacity and report as soon as either starts.
#
# Both EU hosts were full at the time this was written, and capacity appears
# intermittently. Polling by hand wastes wall-clock; this leaves a readable trail
# and exits the moment a pod is RUNNING or startable.
param(
    [string[]]$PodIds = @("qvg4nrmiqb8man", "o4uhh2qw28w9zk"),
    [int]$MaxMinutes = 180,
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = "Continue"

$key = (doppler secrets get RUNPOD_MODEL_TRAINING_API_KEY `
        --project model-training --config dev --plain 2>&1 | Out-String).Trim()
if (-not $key) { Write-Output "DOPPLER_FAILED"; exit 1 }

$headers = @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" }
$deadline = (Get-Date).AddMinutes($MaxMinutes)
$attempt = 0

while ((Get-Date) -lt $deadline) {
    $attempt++
    $stamp = (Get-Date).ToString("HH:mm:ss")

    foreach ($id in $PodIds) {
        try {
            $pod = Invoke-RestMethod -Method Get `
                -Uri "https://rest.runpod.io/v1/pods/$id" -Headers $headers -TimeoutSec 30
            if ($pod.desiredStatus -eq "RUNNING") {
                Write-Output "[$attempt] $stamp $id ALREADY_RUNNING"
                Write-Output "READY=$id"
                exit 0
            }
        }
        catch { }

        try {
            Invoke-RestMethod -Method Post -Uri "https://rest.runpod.io/v1/pods/$id/start" `
                -Headers $headers -TimeoutSec 60 -ErrorAction Stop | Out-Null
            Write-Output "[$attempt] $stamp $id START_ACCEPTED"
        }
        catch {
            Write-Output "[$attempt] $stamp $id full"
        }
    }

    Start-Sleep -Seconds $IntervalSeconds

    foreach ($id in $PodIds) {
        try {
            $pod = Invoke-RestMethod -Method Get `
                -Uri "https://rest.runpod.io/v1/pods/$id" -Headers $headers -TimeoutSec 30
            if ($pod.desiredStatus -eq "RUNNING") {
                Write-Output "READY=$id after $attempt attempt(s)"
                exit 0
            }
        }
        catch { }
    }
}

Write-Output "TIMEOUT after $attempt attempt(s)"
exit 2
