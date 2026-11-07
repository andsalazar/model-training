#!/bin/bash
date -u
echo "--- pid / cpu-time ---"
ps -o pid,etime,time,cmd -p 655 2>/dev/null || echo "pid 655 gone"
echo "--- disk ---"
df -h / | tail -1
echo "--- venv size ---"
du -sh /local/ovi-env
echo "--- setup.log tail ---"
tail -n 10 /root/setup.log
