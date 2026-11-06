# Run a command on the Ovi metric pod over direct SSH.
#
# The Runpod SSH *proxy* hangs on this pod, and its PTY requirement breaks
# non-interactive use. The direct endpoint works without a PTY, so this helper
# targets that and keeps BatchMode on so a missing key fails fast instead of
# prompting.
#
# Usage:
#   .\scripts\ovi_ssh.ps1 -Command "nvidia-smi"
#   .\scripts\ovi_ssh.ps1 -Command "python3 --version" -TimeoutSeconds 60
param(
    [Parameter(Mandatory = $true)]
    [string]$Command,
    [string]$HostName = "213.173.108.44",
    [int]$Port = 36586,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Continue"

$job = Start-Job -ScriptBlock {
    param($cmd, $h, $p, $k)
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes `
        -o ServerAliveInterval=20 -o ServerAliveCountMax=3 `
        -i $k root@$h -p $p $cmd 2>&1
} -ArgumentList $Command, $HostName, $Port, $KeyPath

if (Wait-Job $job -Timeout $TimeoutSeconds) {
    Receive-Job $job
} else {
    Write-Output "TIMEOUT after ${TimeoutSeconds}s"
    Stop-Job $job
}
Remove-Job $job -Force -ErrorAction SilentlyContinue
