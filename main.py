"""
Simple Gait Analysis Main Script

Minimal orchestrator that coordinates detection, pose analysis, and data processing.
Clean separation of concerns with just the essential workflow.
"""

import argparse
import os
import torch

# Import our modular components
from detection_tracking import DetectionTracker
from pose_analysis import PoseAnalyzer
from data_processing import DataProcessor

# Import utilities
from utils.visualization import Visualizer
from utils.video_processor import VideoProcessor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gait Analysis with Modular Architecture")
    
    # Input/Output
    parser.add_argument("--video", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file (optional)")
    parser.add_argument("--results_dir", type=str, default="results", 
                       help="Directory to save all output files")
    
    # Model settings
    parser.add_argument("--use_transreid", action="store_true", default=True,
                       help="Use TransReID for person tracking")
    parser.add_argument("--transreid_model", type=str, default="weights/transreid_vitbase.pth",
                       help="Path to TransReID model weights")
    
    # Tracking parameters
    parser.add_argument("--tracking_iou", type=float, default=0.5,
                       help="IoU threshold for tracking association")
    parser.add_argument("--tracking_age", type=int, default=30,
                       help="Maximum age for tracks before deletion")
    
    # Processing options
    parser.add_argument("--buffer_size", type=float, default=0.05,
                       help="Buffer size ratio around detected person")
    parser.add_argument("--save_bbox_info", action="store_true", default=False,
                       help="Save bounding box information to JSON")
    
    # Post-processing
    parser.add_argument("--merge_ids", action="store_true", default=False,
                       help="Run interactive ID merging after processing")
    
    # Display options
    parser.add_argument("--display", action="store_true", default=False,
                       help="Display video during processing")
    parser.add_argument("--save_video", action="store_true", default=False,
                       help="Save processed video with visualizations")
    
    return parser.parse_args()

def get_device_config():
    """Configure devices for optimal performance"""
    if torch.cuda.is_available():
        device = 'cuda'
        pose_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        pose_device = 'cpu'  # MPS might not support pose detection well
    else:
        device = 'cpu'
        pose_device = 'cpu'
    
    print(f"Using device: {device}, Pose device: {pose_device}")
    return device, pose_device

def main():
    """Main processing pipeline with frame-by-frame feature storage including normalized features"""
    args = parse_args()
    
    # Setup devices
    device, pose_device = get_device_config()
    
    # Initialize modular components
    print("Initializing components...")
    
    # 1. Detection and Tracking
    detector_tracker = DetectionTracker(
        device=device,
        pose_device=pose_device,
        use_transreid=args.use_transreid,
        transreid_model=args.transreid_model,
        tracking_iou=args.tracking_iou,
        tracking_age=args.tracking_age
    )
    
    # 2. Pose Analysis (with feature computation per frame)
    pose_analyzer = PoseAnalyzer(
        model_pose=detector_tracker.model_pose,
        pose_device=pose_device,
        history_length=5
    )
    
    # 3. Data Processing (frame-by-frame storage)
    data_processor = DataProcessor()
    
    # 4. Visualization and Video Processing
    visualizer = Visualizer()
    video_processor = VideoProcessor(
        args.video,
        output_path=args.output if args.save_video else None,
        headless=not args.display
    )
    
    print("Starting video processing...")
    
    # Main processing loop (compute features per frame)
    def process_frame(frame, frame_count, fps):
        """Process each frame and compute features immediately"""
        
        # 1. Detect and track people
        tracked_detections = detector_tracker.detect_and_track(frame)
        
        # 2. Process poses for each detection
        for detection in tracked_detections:
            if not hasattr(detection, 'track_id') or detection.track_id <= 0:
                continue
            
            # Collect bounding box info
            data_processor.collect_bbox_info(detection.track_id, detection.bbox, frame_count)
            
            # Process pose and get features for this frame
            pose_results = pose_analyzer.process_pose(frame, detection, frame_count)
            
            # 🔥 Calculate features for this specific frame
            if pose_results:
                # Get the latest keypoints for this track
                features_dict = pose_analyzer.calculate_frame_features(detection.track_id, frame_count)
                
                # Store frame-by-frame features
                if features_dict:
                    data_processor.store_frame_features(detection.track_id, frame_count, features_dict)
                
                # Visualize results
                color = visualizer.get_color(detection.track_id)
                
                for pose_result in pose_results:
                    keypoints = pose_result['keypoints']
                    crop_offset = pose_result['crop_offset']
                    
                    # Draw bounding box and ID
                    visualizer.draw_bbox(frame, detection.bbox, detection.track_id, color)
                    
                    # Draw keypoints and skeleton
                    visualizer.draw_keypoints(frame, keypoints, crop_offset[0], crop_offset[1], color)
        
        return frame
    
    # Process the video (compute features per frame)
    video_processor.process_video(process_frame)
    
    print("Video processing complete. Frame-by-frame features collected.")
    
    # 🔥 Step 1: Check collected data
    stats = data_processor.get_summary_stats()
    print(f"Collected features for {stats['total_tracks']} tracks")
    print(f"Total feature records: {stats['total_feature_records']}")
    
    # 🔥 Step 2: Run ID merger on frame features (if requested)
    merged_ids = None
    id_to_name = None
    
    if args.merge_ids:
        print("\n=== Starting ID Merging ===")
        temp_paths = data_processor.export_temp_data_for_merger(args)
        merged_ids, id_to_name = data_processor.merge_ids(args, temp_paths)
        print(f"ID merging complete: {merged_ids}")
    
    # 🔥 Step 3: Apply ID merging to frame features
    print("\n=== Applying ID Merging to Frame Features ===")
    merged_frame_features = data_processor.apply_id_merging_to_frame_features(merged_ids or {})
    
    # 🔥 Step 4: Export frame-by-frame features WITH normalized gait features
    print("\n=== Exporting Frame-by-Frame Features with Normalized Features ===")
    final_paths = data_processor.export_frame_by_frame_features(
        merged_frame_features, 
        args, 
        id_to_name, 
        pose_analyzer  # 🔥 Pass pose_analyzer to include normalized features
    )
    
    # 🔥 Step 5: Export additional normalized features summary (optional)
    if pose_analyzer and hasattr(pose_analyzer, 'gait_analyzer') and pose_analyzer.gait_analyzer:
        print("\n=== Exporting Normalized Features Summary ===")
        normalized_summary_path = os.path.join(args.results_dir, "normalized_features_summary.csv")
        try:
            pose_analyzer.export_features_csv(normalized_summary_path)
            print(f"✓ Normalized features summary saved to: {normalized_summary_path}")
        except Exception as e:
            print(f"Note: Could not export normalized summary: {e}")
    
    print("\n=== Processing Complete ===")
    print(f"Results saved to: {args.results_dir}")
    print(f"Frame-by-frame CSV with normalized features: {final_paths['features_csv']}")
    
    # Summary statistics
    print(f"\nFinal Summary:")
    print(f"- Original tracks: {stats['total_tracks']}")
    print(f"- Final tracks after merging: {len(merged_frame_features)}")
    print(f"- Total feature records: {stats['total_feature_records']}")
    
    if merged_ids:
        print(f"- ID merges applied: {len(merged_ids)}")
        for old_id, new_id in merged_ids.items():
            print(f"  ID {old_id} → ID {new_id}")
    
    for track_id in sorted(merged_frame_features.keys()):
        frames_count = len(merged_frame_features[track_id])
        print(f"  Final Track {track_id}: {frames_count} frames")
    
    # 🔥 Feature information
    if pose_analyzer and hasattr(pose_analyzer, 'gait_analyzer') and pose_analyzer.gait_analyzer:
        print(f"\nFeature Details:")
        sample_track = next(iter(merged_frame_features.keys()))
        sample_features = pose_analyzer.gait_analyzer.get_features(sample_track)
        sample_invariant = pose_analyzer.gait_analyzer.calculate_view_invariant_features(sample_track)
        
        if sample_features:
            print(f"- Normalized features per track: {len(sample_features)}")
        if sample_invariant:
            print(f"- View-invariant features per track: {len(sample_invariant)}")
        
        # Show some feature names
        if sample_features:
            feature_names = list(sample_features.keys())[:5]
            print(f"- Sample normalized features: {feature_names}")
        if sample_invariant:
            invariant_names = list(sample_invariant.keys())[:5]
            print(f"- Sample view-invariant features: {invariant_names}")

if __name__ == "__main__":
    main()