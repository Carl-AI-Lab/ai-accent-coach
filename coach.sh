#!/bin/bash
# AccentCoach service manager
DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/.accent-coach.pid"
LOG_FILE="$DIR/accent-coach.log"

# GPU users: uncomment and set the real paths below.
# To find them, run:
#   python3 -c "import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0]); print(nvidia.cudnn.lib.__path__[0])"
# export LD_LIBRARY_PATH="/path/to/nvidia/cublas/lib:/path/to/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# China mainland users: uncomment to use HuggingFace mirror
# export HF_ENDPOINT="https://hf-mirror.com"

case "$1" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "AccentCoach is already running (PID $(cat "$PID_FILE"))"
      exit 0
    fi
    echo "Starting AccentCoach..."
    cd "$DIR"
    nohup python3 app.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "AccentCoach started (PID $!) — https://localhost:8443"
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping AccentCoach (PID $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Stopped."
      else
        echo "PID $PID not running. Cleaning up."
        rm -f "$PID_FILE"
      fi
    else
      echo "AccentCoach is not running."
    fi
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" start
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "AccentCoach is running (PID $(cat "$PID_FILE"))"
    else
      echo "AccentCoach is not running."
    fi
    ;;
  log)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|log}"
    exit 1
    ;;
esac
