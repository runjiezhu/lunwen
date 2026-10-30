# Restart the training console once the in-flight merged job finishes.
#
# The server keeps the merged trainer as a child process and streams its
# stdout through a pipe, so killing the server mid-run would break that pipe
# and abort the job. This waits for the child to exit, then restarts the
# server so it picks up the current server.py (--iters / --resume passthrough).

$ErrorActionPreference = 'SilentlyContinue'

$root = 'C:\Users\zhuru\yuelaogame\lunwen'
$jobPid = 44772      # merged trainer child
$serverPid = 444     # console server
$log = Join-Path $root 'console\restart_server.log'

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
}

Log "watcher started (waiting for job pid $jobPid)"

while (Get-Process -Id $jobPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}

Log "merged job $jobPid exited; restarting console server"

if (Get-Process -Id $serverPid -ErrorAction SilentlyContinue) {
    Stop-Process -Id $serverPid -Force
    Start-Sleep -Seconds 3
}

Start-Process -FilePath 'C:\Program Files\Python311\python.exe' `
    -ArgumentList 'console\server.py', '--port', '8770' `
    -WorkingDirectory $root -WindowStyle Hidden

Start-Sleep -Seconds 4
Log "console server restarted"
