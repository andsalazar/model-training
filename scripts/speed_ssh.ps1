# Run a command on the Ovi speed pod over direct SSH.
#
# The Runpod SSH proxy hangs on these pods and its PTY requirement breaks
# non-interactive use. The direct endpoint works with an explicit key and no PTY.
#
# The command is passed as an argument to ssh rather than piped through stdin.
# Piping from PowerShell sends CRLF, and the remote shell then sees a trailing \r
# on every command ("head: cannot open 'file'$'\r'"). An argument avoids stdin.
#
# Usage:
#   .\scripts\speed_ssh.ps1 -Command "nvidia-smi"
#   .\scripts\speed_ssh.ps1 -Command "ls /workspace" -TimeoutSeconds 300

param(
    [Parameter(Mandatory = $true)]
    [string]$Command,
    [int]$TimeoutSeconds = 120,
    [string]$SshHost = "157.157.221.29",
    [int]$Port = 11634
)

$key = "$env:USERPROFILE\.ssh\id_ed25519"
$job = Start-Job -ScriptBlock {
    param($cmd, $h, $p, $k)
    & ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes `
        -i $k -p $p "root@$h" $cmd 2>&1
} -ArgumentList $Command, $SshHost, $Port, $key

if (Wait-Job $job -Timeout $TimeoutSeconds) {
    Receive-Job $job
} else {
    Write-Output "TIMEOUT after ${TimeoutSeconds}s"
    Stop-Job $job
}
Remove-Job $job -Force -ErrorAction SilentlyContinue
