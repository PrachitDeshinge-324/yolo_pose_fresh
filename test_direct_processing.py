# test_direct_processing.py
import cv2
from detection_tracking import DetectionTracker
from pose_analysis import PoseAnalyzer
from utils.visualization import Visualizer

def test_direct_processing():
    """Test detection and pose processing directly without VideoProcessor"""
    
    print("Testing direct frame processing...")
    
    # Initialize components
    detector_tracker = DetectionTracker(
        device='mps',
        pose_device='cpu',
        use_transreid=False,
        tracking_iou=0.5,
        tracking_age=30
    )
    
    pose_analyzer = PoseAnalyzer(
        model_pose=detector_tracker.model_pose,
        pose_device='cpu',
        history_length=5
    )
    
    visualizer = Visualizer()
    
    # Open video directly
    cap = cv2.VideoCapture("../Person_New/input/3c1.mp4")
    
    frame_count = 0
    total_tracks_found = 0
    
    while frame_count < 180:  # Test first 180 frames (6 seconds)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame for testing
        if frame_count % 30 == 0:
            print(f"\n=== Frame {frame_count} ===")
            
            # 1. Detection
            tracked_detections = detector_tracker.detect_and_track(frame)
            print(f"Found {len(tracked_detections)} detections")
            
            for detection in tracked_detections:
                if hasattr(detection, 'track_id') and detection.track_id > 0:
                    track_id = detection.track_id
                    total_tracks_found += 1
                    
                    print(f"  Processing track {track_id}")
                    
                    # 2. Pose processing
                    pose_results = pose_analyzer.process_pose(frame, detection, frame_count)
                    
                    if pose_results:
                        print(f"    ✅ Pose: {len(pose_results)} results")
                        
                        # 3. Feature extraction
                        features_dict = pose_analyzer.calculate_frame_features(track_id, frame_count)
                        
                        if features_dict:
                            print(f"    ✅ Features: {len(features_dict)} extracted")
                            print(f"      Sample: {list(features_dict.keys())[:3]}...")
                        else:
                            print(f"    ❌ No features extracted")
                    else:
                        print(f"    ❌ No pose results")
                    
                    # 4. Visualization
                    color = visualizer.get_color(track_id)
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Track {track_id}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show frame
            cv2.imshow("Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Test Summary ===")
    print(f"Frames processed: {frame_count}")
    print(f"Total tracks found: {total_tracks_found}")
    
    if total_tracks_found > 0:
        print("✅ Detection and pose processing working!")
    else:
        print("❌ No tracks found - investigate detection/pose pipeline")

if __name__ == "__main__":
    test_direct_processing()