#!/bin/bash

# Real-time Gait Recognition Inference
echo "=== Real-time Gait Recognition Inference ==="

# Set parameters
VIDEO_FILE="../Person_New/input/3c1.mp4"
MODEL_FILE="lstm_gait_results/best_model.pth"
OUTPUT_VIDEO="inference_results/identified_gait.mp4"
RESULTS_DIR="inference_results"

# Create results directory
mkdir -p $RESULTS_DIR

echo "🎯 Running inference on video: $VIDEO_FILE"
echo "📱 Using trained model: $MODEL_FILE"

# Run inference with basic YOLO detection (disable TransReID)
python gait_inference.py \
    --video $VIDEO_FILE \
    --model $MODEL_FILE \
    --output $OUTPUT_VIDEO \
    --results_dir $RESULTS_DIR \
    --sequence_length 20 \
    --min_confidence 0.7 \
    --min_predictions 3 \
    --display \
    --save_video
    # 🔥 Removed --use_transreid flag (defaults to False now)

echo "✅ Inference completed! Check results in $RESULTS_DIR"