#!/bin/bash

# Training Monitor Script
# Checks training progress every 10 minutes during LR finding and first phase
# Then every 30 minutes during main training

LOG_FILE="training.log"
CSV_FILE="training_results.csv"
CHECK_INTERVAL_INITIAL=600  # 10 minutes in seconds
CHECK_INTERVAL_TRAINING=1800  # 30 minutes in seconds

echo "=========================================="
echo "Training Monitor Started"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "Results file: $CSV_FILE"
echo ""

# Check if training is running
check_training_status() {
    if pgrep -f "train_models.py" > /dev/null; then
        return 0  # Training is running
    else
        return 1  # Training is not running
    fi
}

# Get latest progress from log
show_progress() {
    echo ""
    echo "=========================================="
    echo "Training Progress - $(date)"
    echo "=========================================="

    # Check GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s: %s | Temp: %s°C | GPU Util: %s%% | Mem Util: %s%% | Mem: %sMB/%sMB\n", $1, $2, $3, $4, $5, $6, $7}'
        echo ""
    fi

    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Recent Log Output:"
        echo "------------------------------------------"
        tail -n 30 "$LOG_FILE"
        echo ""
    fi

    if [ -f "$CSV_FILE" ]; then
        echo ""
        echo "Training Results Summary:"
        echo "------------------------------------------"
        # Show last 10 epochs for each model
        python3 << 'PYTHON'
import pandas as pd
try:
    df = pd.read_csv('training_results.csv')
    if len(df) > 0:
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model].tail(10)
            print(f"\n{model}:")
            print(f"  Latest epoch: {model_df['epoch'].iloc[-1]}")
            print(f"  Latest accuracy: {model_df['accuracy'].iloc[-1]:.4f}")
            print(f"  Best accuracy: {df[df['model_name'] == model]['accuracy'].max():.4f}")
            print(f"  Latest loss: {model_df['valid_loss'].iloc[-1]:.4f}")
    else:
        print("No results yet")
except Exception as e:
    print(f"Could not read results: {e}")
PYTHON
        echo ""
    fi
}

# Wait for training to start
echo "Waiting for training to start..."
while ! check_training_status; do
    sleep 5
done

echo "Training process detected!"
echo ""

# Determine phase based on epoch count
phase_count=0
last_epoch=0

while check_training_status; do
    show_progress

    # Check which phase we're in based on epoch count
    if [ -f "$CSV_FILE" ]; then
        current_epoch=$(tail -1 "$CSV_FILE" | cut -d',' -f3 2>/dev/null)
        if [ ! -z "$current_epoch" ] && [ "$current_epoch" != "epoch" ]; then
            if [ $current_epoch -gt $last_epoch ]; then
                last_epoch=$current_epoch
                if [ $current_epoch -lt 15 ]; then
                    # Initial phases - check every 10 minutes
                    INTERVAL=$CHECK_INTERVAL_INITIAL
                    echo "Phase: Initial training (checking every 10 min)"
                else
                    # Main training - check every 30 minutes
                    INTERVAL=$CHECK_INTERVAL_TRAINING
                    echo "Phase: Main training (checking every 30 min)"
                fi
            fi
        else
            INTERVAL=$CHECK_INTERVAL_INITIAL
        fi
    else
        INTERVAL=$CHECK_INTERVAL_INITIAL
    fi

    echo ""
    echo "Next check in $(($INTERVAL / 60)) minutes..."
    sleep $INTERVAL
done

echo ""
echo "=========================================="
echo "Training Completed or Stopped"
echo "=========================================="
echo "End time: $(date)"
show_progress
