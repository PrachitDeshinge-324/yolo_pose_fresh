"""
Enhanced Gait Analysis Main Script with YOLO11 Silhouette Extraction and OpenGait Features

Combines pose-based features with silhouette-based OpenGait embeddings for robust person identification.
"""

import argparse
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Import our modular components
from detection_tracking import DetectionTracker
from pose_analysis import PoseAnalyzer
from data_processing import DataProcessor

# Import utilities
from utils.visualization import Visualizer
from utils.video_processor import VideoProcessor
from utils.opengait import OpenGaitEmbedder

class SilhouetteExtractor:
    """YOLO11-based silhouette extraction for OpenGait features"""
    
    def __init__(self, model_path="yolo11n-seg.pt", device="cpu"):
        """
        Initialize YOLO11 segmentation model for silhouette extraction
        
        Args:
            model_path: Path to YOLO11 segmentation model
            device: Device to run inference on
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"YOLO11 segmentation model loaded on {device}")
    
    def extract_person_silhouette(self, frame, bbox, target_size=(64, 128)):
        """
        Extract person silhouette from frame using YOLO11 segmentation
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Target size for silhouette (width, height)
            
        Returns:
            silhouette: Binary silhouette mask or None if extraction failed
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expand bbox slightly for better segmentation
            h, w = frame.shape[:2]
            expansion = 0.1
            dx = int((x2 - x1) * expansion)
            dy = int((y2 - y1) * expansion)
            
            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)
            
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
            
            # Run YOLO11 segmentation
            results = self.model(person_crop, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            
            # Find person class (class 0 in COCO)
            if result.masks is None:
                return None
            
            person_masks = []
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == 0:  # Person class
                    person_masks.append(result.masks.data[i].cpu().numpy())
            
            if not person_masks:
                return None
            
            # Combine all person masks
            combined_mask = np.zeros_like(person_masks[0])
            for mask in person_masks:
                combined_mask = np.logical_or(combined_mask, mask > 0.5)
            
            # Resize to target size
            combined_mask = combined_mask.astype(np.uint8) * 255
            silhouette = cv2.resize(combined_mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel)
            
            return silhouette
            
        except Exception as e:
            print(f"Error extracting silhouette: {e}")
            return None
    
    def extract_silhouette_sequence(self, frame_sequence, bbox_sequence, target_size=(64, 128)):
        """
        Extract silhouette sequence for temporal modeling
        
        Args:
            frame_sequence: List of frames
            bbox_sequence: List of bounding boxes
            target_size: Target silhouette size
            
        Returns:
            silhouettes: List of silhouette masks
        """
        silhouettes = []
        
        for frame, bbox in zip(frame_sequence, bbox_sequence):
            silhouette = self.extract_person_silhouette(frame, bbox, target_size)
            if silhouette is not None:
                silhouettes.append(silhouette)
        
        return silhouettes if len(silhouettes) > 0 else None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Gait Analysis with OpenGait Features")
    
    # Input/Output
    parser.add_argument("--video",default="../Person_New/input/3c.mp4" , help="Path to input video file")
    parser.add_argument("--output",default="results/3c.mp4", help="Path to output video file (optional)")
    parser.add_argument("--results_dir", type=str, default="results", 
                       help="Directory to save all output files")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Start processing from this frame number")
    parser.add_argument("--end_frame", type=int, default=2000,
                       help="End processing at this frame number (0 for full video)")
    
    # Model settings
    parser.add_argument("--use_transreid", action="store_true", default=True,
                       help="Use TransReID for person tracking")
    parser.add_argument("--transreid_model", type=str, default="weights/transreid_vitbase.pth",
                       help="Path to TransReID model weights")
    
    # YOLO11 and OpenGait settings
    parser.add_argument("--yolo_model", type=str, default="weights/yolo11x-seg.pt",
                       help="Path to YOLO11 segmentation model")
    parser.add_argument("--opengait_weights", type=str, default="weights/GaitBase_DA-60000.pt",
                       help="Path to OpenGait model weights")
    parser.add_argument("--silhouette_size", type=int, nargs=2, default=[64, 128],
                       help="Target silhouette size [width, height]")
    parser.add_argument("--sequence_length", type=int, default=30,
                       help="Sequence length for OpenGait temporal modeling")
    
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
    parser.add_argument("--merge_ids", action="store_true", default=True,
                       help="Run interactive ID merging after processing")
    
    # Display options
    parser.add_argument("--display", action="store_true", default=True,
                       help="Display video during processing")
    parser.add_argument("--save_video", action="store_true", default=True,
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
    """Enhanced main processing pipeline with OpenGait silhouette features"""
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
    
    # 3. Silhouette Extraction with YOLO11
    print("Loading YOLO11 segmentation model...")
    silhouette_extractor = SilhouetteExtractor(
        model_path=args.yolo_model,
        device=device
    )
    
    # 4. OpenGait Embedder
    print("Loading OpenGait embedder...")
    opengait_embedder = OpenGaitEmbedder(
        weights_path=args.opengait_weights,
        device=device
    )
    
    # 5. Data Processing (enhanced for silhouette features)
    data_processor = DataProcessor()
    
    # 6. Visualization and Video Processing
    visualizer = Visualizer()
    video_processor = VideoProcessor(
        args.video,
        output_path=args.output if args.save_video else None,
        headless=not args.display,
        start_frame=args.start_frame if hasattr(args, 'start_frame') else 0,
        end_frame=args.end_frame if hasattr(args, 'end_frame') else 2000,
    )
    
    print("Starting enhanced video processing with OpenGait features...")
    
    # Storage for silhouette sequences
    silhouette_sequences = {}  # track_id -> list of (frame_num, frame, bbox, silhouette)
    opengait_features = {}     # track_id -> list of embeddings
    
    # Main processing loop
    def process_frame(frame, frame_count, fps):
        """Process each frame with both pose and silhouette features"""
        
        # 1. Detect and track people
        tracked_detections = detector_tracker.detect_and_track(frame)
        
        # 2. Process each detection
        for detection in tracked_detections:
            if not hasattr(detection, 'track_id') or detection.track_id is None or detection.track_id <= 0:
                continue
            
            track_id = detection.track_id
            
            # Collect bounding box info
            data_processor.collect_bbox_info(track_id, detection.bbox, frame_count)
            
            # 2a. Process pose and get traditional features
            pose_results = pose_analyzer.process_pose(frame, detection, frame_count)
            
            if pose_results:
                # Calculate traditional pose features
                features_dict = pose_analyzer.calculate_frame_features(track_id, frame_count)
                
                # if features_dict:
                #     data_processor.store_frame_features(track_id, frame_count, features_dict)
            
            # 2b. Extract silhouette and add to sequence
            silhouette = silhouette_extractor.extract_person_silhouette(
                frame, 
                detection.bbox, 
                target_size=tuple(args.silhouette_size)
            )
            
            if silhouette is not None:
                # Initialize sequence storage for new tracks
                if track_id not in silhouette_sequences:
                    silhouette_sequences[track_id] = []
                    opengait_features[track_id] = []
                
                # Add to sequence
                silhouette_sequences[track_id].append({
                    'frame_num': frame_count,
                    'frame': frame.copy(),
                    'bbox': detection.bbox,
                    'silhouette': silhouette
                })
                
                # Extract OpenGait features when we have enough frames
                sequence_data = silhouette_sequences[track_id]
                if len(sequence_data) >= args.sequence_length:
                    # Get recent silhouettes
                    recent_silhouettes = [item['silhouette'] for item in sequence_data[-args.sequence_length:]]
                    
                    try:
                        # Extract OpenGait embedding
                        gait_embedding = opengait_embedder.extract(
                            recent_silhouettes, 
                            use_temporal_aggregation=True
                        )
                        
                        # Store OpenGait features
                        opengait_features[track_id].append({
                            'frame_num': frame_count,
                            'embedding': gait_embedding,
                            'sequence_length': len(recent_silhouettes)
                        })
                        
                        # Also store in data processor for CSV export
                        opengait_dict = {
                            f'opengait_dim_{i}': float(gait_embedding[i]) 
                            for i in range(len(gait_embedding))
                        }
                        opengait_dict['opengait_embedding_norm'] = float(np.linalg.norm(gait_embedding))
                        
                        # Combine with existing features or create new entry
                        if hasattr(data_processor, 'frame_features') and track_id in data_processor.frame_features:
                            if frame_count in data_processor.frame_features[track_id]:
                                data_processor.frame_features[track_id][frame_count].update(opengait_dict)
                            else:
                                data_processor.store_frame_features(track_id, frame_count, opengait_dict)
                        else:
                            data_processor.store_frame_features(track_id, frame_count, opengait_dict)
                        
                    except Exception as e:
                        print(f"Error extracting OpenGait features for track {track_id}: {e}")
                
                # Limit sequence length to prevent memory issues
                if len(silhouette_sequences[track_id]) > args.sequence_length * 2:
                    silhouette_sequences[track_id] = silhouette_sequences[track_id][-args.sequence_length:]
            
            # 3. Visualization
            if pose_results:
                color = visualizer.get_color(track_id)
                
                for pose_result in pose_results:
                    keypoints = pose_result['keypoints']
                    crop_offset = pose_result['crop_offset']
                    
                    # Draw bounding box and ID
                    bbox_text = f"ID:{track_id}"
                    
                    # Add OpenGait info if available
                    if track_id in opengait_features and len(opengait_features[track_id]) > 0:
                        latest_features = len(opengait_features[track_id])
                        bbox_text += f" G:{latest_features}"
                    
                    visualizer.draw_bbox(frame, detection.bbox, track_id, color)
                    
                    # Draw keypoints and skeleton
                    visualizer.draw_keypoints(frame, keypoints, crop_offset[0], crop_offset[1], color)
                    
                    # Optionally draw silhouette indicator
                    if silhouette is not None:
                        # Small green dot to indicate silhouette extraction
                        cv2.circle(frame, (int(detection.bbox[0] + 10), int(detection.bbox[1] + 10)), 
                                 3, (0, 255, 0), -1)
        
        return frame
    
    # Process the video
    video_processor.process_video(process_frame)
    
    print("Video processing complete. Features collected.")
    
    # Print OpenGait statistics
    total_opengait_features = sum(len(features) for features in opengait_features.values())
    print(f"OpenGait Features Summary:")
    print(f"- Tracks with silhouettes: {len(silhouette_sequences)}")
    print(f"- Tracks with OpenGait features: {len(opengait_features)}")
    print(f"- Total OpenGait embeddings: {total_opengait_features}")
    
    for track_id in sorted(opengait_features.keys()):
        embeddings_count = len(opengait_features[track_id])
        silhouettes_count = len(silhouette_sequences.get(track_id, []))
        print(f"  Track {track_id}: {embeddings_count} embeddings from {silhouettes_count} silhouettes")
    
    # Standard processing continues...
    stats = data_processor.get_summary_stats()
    print(f"Collected features for {stats['total_tracks']} tracks")
    print(f"Total feature records: {stats['total_feature_records']}")
    
    # ID merging
    merged_ids = None
    id_to_name = None
    
    if args.merge_ids:
        print("\n=== Starting ID Merging ===")
        temp_paths = data_processor.export_temp_data_for_merger(args)
        merged_ids, id_to_name = data_processor.merge_ids(args, temp_paths)
        print(f"ID merging complete: {merged_ids}")
    
    # Apply ID merging to frame features
    print("\n=== Applying ID Merging to Frame Features ===")
    merged_frame_features = data_processor.apply_id_merging_to_frame_features(merged_ids or {})
    
    # Export enhanced features
    print("\n=== Exporting Enhanced Features with OpenGait ===")
    final_paths = data_processor.export_frame_by_frame_features(
        merged_frame_features, 
        args, 
        id_to_name, 
        pose_analyzer
    )
    
    # Export OpenGait embeddings separately
    opengait_export_path = os.path.join(args.results_dir, "opengait_embeddings.json")
    try:
        import json
        
        # Apply ID merging to OpenGait features
        merged_opengait = {}
        for track_id, features in opengait_features.items():
            final_id = merged_ids.get(track_id, track_id) if merged_ids else track_id
            if final_id not in merged_opengait:
                merged_opengait[final_id] = []
            merged_opengait[final_id].extend(features)
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for track_id, features in merged_opengait.items():
            export_data[str(track_id)] = []
            for feature in features:
                export_data[str(track_id)].append({
                    'frame_num': feature['frame_num'],
                    'embedding': feature['embedding'].tolist(),
                    'sequence_length': feature['sequence_length']
                })
        
        with open(opengait_export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ OpenGait embeddings saved to: {opengait_export_path}")
        
    except Exception as e:
        print(f"Error exporting OpenGait embeddings: {e}")
    
    print("\n=== Processing Complete ===")
    print(f"Results saved to: {args.results_dir}")
    print(f"Enhanced CSV with OpenGait features: {final_paths['features_csv']}")
    print(f"OpenGait embeddings: {opengait_export_path}")
    
    # Final summary
    print(f"\nFinal Summary:")
    print(f"- Original tracks: {stats['total_tracks']}")
    print(f"- Final tracks after merging: {len(merged_frame_features)}")
    print(f"- Total feature records: {stats['total_feature_records']}")
    print(f"- OpenGait embeddings: {total_opengait_features}")
    
    if merged_ids:
        print(f"- ID merges applied: {len(merged_ids)}")

if __name__ == "__main__":
    main()