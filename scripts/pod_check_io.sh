#!/bin/bash
# Report whether a pip install is still moving, by sampling its read counter.
PID="${1:-404}"
if [ ! -d "/proc/$PID" ]; then
  echo "PID $PID is gone"
  exit 0
fi
A=$(awk '/^rchar/ {print $2}' "/proc/$PID/io")
sleep 30
B=$(awk '/^rchar/ {print $2}' "/proc/$PID/io")
echo "rchar_delta=$((B - A)) bytes in 30s"
echo "state=$(awk '{print $3}' "/proc/$PID/stat")"
echo "---log tail---"
tail -2 /workspace/setup.log
