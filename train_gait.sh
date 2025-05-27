#!/bin/bash

# Train Bidirectional LSTM for Gait Recognition (Mac Optimized)
echo "=== Training Bidirectional LSTM for Gait Recognition (Mac Optimized) ==="

# Set parameters optimized for Mac and small datasets
CSV_FILE="results1c/1c_frame_features.csv"  # Your CSV file
OUTPUT_DIR="results1c/lstm_gait_results"
SEQUENCE_LENGTH=20  # Reduced to create more sequences
HIDDEN_SIZE=64      # Smaller model to prevent overfitting
NUM_EPOCHS=30       # Fewer epochs for small datasets
BATCH_SIZE=4        # Smaller batch size for better GPU utilization

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🍎 Mac-optimized parameters:"
echo "  - Sequence length: $SEQUENCE_LENGTH (reduced)"
echo "  - Hidden size: $HIDDEN_SIZE (smaller model)"
echo "  - Batch size: $BATCH_SIZE (optimized for MPS)"
echo "  - Epochs: $NUM_EPOCHS (reduced for small dataset)"

# Run training
python train_lstm_gait.py \
    --data $CSV_FILE \
    --output_dir $OUTPUT_DIR \
    --sequence_length $SEQUENCE_LENGTH \
    --step_size 3 \
    --hidden_size $HIDDEN_SIZE \
    --num_layers 2 \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate 0.0005 \
    --dropout 0.4 \
    --min_frames_per_person 30 \
    --patience 10

echo "Training completed! Check results in $OUTPUT_DIR"