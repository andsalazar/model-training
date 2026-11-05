# Run a command on the persistent pod over SSH.
#
# The Runpod SSH proxy ignores a command argument and always drops into an
# interactive shell, so commands are piped through stdin with a PTY. A marker is
# echoed at the end so the caller can tell the command finished rather than the
# connection dropping mid-run.
param(
    [Parameter(Mandatory = $true)]
    [string]$Command,
    [string]$Pod = "o4uhh2qw28w9zk-644117ea@ssh.runpod.io",
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Continue"

$wrapped = "$Command; echo __POD_EXIT__`$?"
$wrapped | ssh -tt -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=30 $Pod 2>&1 |
    ForEach-Object { $_ }
