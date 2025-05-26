echo "Extracting database information..."
python main.py \
  --video "/content/drive/MyDrive/datasets/My Movie.mp4" \
  --output "/content/drive/MyDrive/datasets/results3/3c.mp4" \
  --results_dir "/content/drive/MyDrive/datasets/results3" \
  --save_bbox_info \
  --merge_ids \
  --use_transreid \
  --save_video \
  --start_frame 0 \
  --end_frame 2000 
