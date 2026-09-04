#!/bin/bash

echo "Starting Gazebo..."
gz sim -r -s -v2 "$1" &

PID=$!
echo $PID > /tmp/gzserver.pid

echo "Stored PID: $PID"

wait
