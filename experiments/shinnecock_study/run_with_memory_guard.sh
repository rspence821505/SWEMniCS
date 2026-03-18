#!/bin/bash
# Memory-guarded test runner for Phase 1
# Monitors MPI process memory and kills if exceeding threshold

MAX_RSS_MB=6000  # 6 GB total across all MPI processes
CHECK_INTERVAL=10  # seconds between checks
LOG_FILE="/tmp/phase1_memguard.log"
OUTPUT_FILE="/tmp/phase1_output.log"

echo "Starting Phase 1 test with memory guard (max ${MAX_RSS_MB}MB)" | tee "$LOG_FILE"
echo "Output: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

# Start the test in background
cd /Users/rylanspence/Desktop/Git/DC/Thesis/SWEMniCS/experiments/shinnecock_study
mpirun -n 2 python run_comparison.py --phase 1 --adios-file ../../data/shinnecock_inlet > "$OUTPUT_FILE" 2>&1 &
MPI_PID=$!

echo "MPI launch PID: $MPI_PID" | tee -a "$LOG_FILE"

# Wait a moment for processes to start
sleep 5

while kill -0 $MPI_PID 2>/dev/null; do
    # Get RSS of all python processes spawned by mpirun (in KB on macOS)
    TOTAL_RSS_KB=0
    PROCESS_INFO=""

    for pid in $(pgrep -P $MPI_PID python 2>/dev/null; echo $MPI_PID); do
        if kill -0 $pid 2>/dev/null; then
            RSS_KB=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
            if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt 0 ] 2>/dev/null; then
                TOTAL_RSS_KB=$((TOTAL_RSS_KB + RSS_KB))
                RSS_MB=$((RSS_KB / 1024))
                PROCESS_INFO="$PROCESS_INFO PID=$pid:${RSS_MB}MB"
            fi
        fi
    done

    # Also check for any python processes running run_comparison
    for pid in $(pgrep -f "run_comparison.py" 2>/dev/null); do
        if ! echo "$PROCESS_INFO" | grep -q "PID=$pid"; then
            RSS_KB=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
            if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt 0 ] 2>/dev/null; then
                TOTAL_RSS_KB=$((TOTAL_RSS_KB + RSS_KB))
                RSS_MB=$((RSS_KB / 1024))
                PROCESS_INFO="$PROCESS_INFO PID=$pid:${RSS_MB}MB"
            fi
        fi
    done

    TOTAL_RSS_MB=$((TOTAL_RSS_KB / 1024))
    TIMESTAMP=$(date '+%H:%M:%S')

    echo "[$TIMESTAMP] Total RSS: ${TOTAL_RSS_MB}MB / ${MAX_RSS_MB}MB |$PROCESS_INFO" | tee -a "$LOG_FILE"

    # Check threshold
    if [ "$TOTAL_RSS_MB" -gt "$MAX_RSS_MB" ]; then
        echo "[$TIMESTAMP] MEMORY LIMIT EXCEEDED! Killing MPI processes..." | tee -a "$LOG_FILE"
        # Kill the MPI processes
        pkill -P $MPI_PID 2>/dev/null
        kill $MPI_PID 2>/dev/null
        # Also kill any remaining run_comparison processes
        pkill -f "run_comparison.py" 2>/dev/null
        echo "[$TIMESTAMP] Processes killed. Last output:" | tee -a "$LOG_FILE"
        tail -20 "$OUTPUT_FILE" | tee -a "$LOG_FILE"
        exit 1
    fi

    sleep $CHECK_INTERVAL
done

echo "--- Test completed normally ---" | tee -a "$LOG_FILE"
echo "Exit code: $(wait $MPI_PID; echo $?)" | tee -a "$LOG_FILE"
echo "Last 30 lines of output:" | tee -a "$LOG_FILE"
tail -30 "$OUTPUT_FILE" | tee -a "$LOG_FILE"
