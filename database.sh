echo "Extracting database information..."
python main.py \
  --video "../Person_New/input/1c.mp4" \
  --output "results1c/1c.mp4" \
  --results_dir "results1c" \
  --save_bbox_info \
  --merge_ids \
  --use_transreid \
  --save_video \
  --start_frame 0 \
  --end_frame 600 
