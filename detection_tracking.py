"""
Detection and Tracking Module

Handles YOLO person detection and TransReID-based tracking.
All detection and tracking functionality consolidated here.
"""

from ultralytics import YOLO
import cv2 as cv
import torch
from filterpy.kalman import KalmanFilter
import numpy as np
import logging
from utils.trackers import YOLOTracker, Detection
from utils.transreid_model import TransReIDModel, load_transreid_model

# Suppress Ultralytics YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class TransReIDTracker:
    """TransReID-based person tracker with robust re-identification"""
    
    def __init__(self, device='cpu', reid_model_path='weights/vit_small_cfs_lup.pth', 
                 iou_threshold=0.5, conf_threshold=0.5, max_age=30):
        self.device = device
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.feature_history = {}
        self.max_history = 10
        self.kalman_filters = {}
        self.track_confidence = {}  # Track confidence scores
        self.confirmed_threshold = 3  # Minimum detections to confirm a track
        self.pending_tracks = {}  # Tracks waiting to be confirmed

        # Load TransReID model
        try:
            self.reid_model = load_transreid_model(reid_model_path, device)
            print(f"TransReID model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading TransReID model: {e}")
            self.reid_model = None
        
        self.feature_db = {}
    
    def extract_feature(self, image_crop):
        """Extract feature vector from person crop using TransReID"""
        if self.reid_model is None or image_crop.size == 0:
            return None
            
        try:
            features = self.reid_model.extract_features(image_crop)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return max(0.0, min(iou, 1.0))
    
    # Create Kalman filter for new tracks
    def initialize_kalman(self, bbox):
        """Initialize Kalman filter for a new track"""
        kf = KalmanFilter(dim_x=8, dim_z=4)  # State: [x, y, w, h, vx, vy, vw, vh], Measurement: [x, y, w, h]
        
        # Initial state
        x1, y1, x2, y2 = bbox
        w, h = x2-x1, y2-y1
        kf.x = np.array([[x1], [y1], [w], [h], [0], [0], [0], [0]])
        
        # State transition matrix (constant velocity model)
        kf.F = np.eye(8)
        kf.F[0, 4] = 1.0  # x += vx
        kf.F[1, 5] = 1.0  # y += vy
        kf.F[2, 6] = 1.0  # w += vw
        kf.F[3, 7] = 1.0  # h += vh
        
        # Measurement matrix (we only observe x, y, w, h)
        kf.H = np.zeros((4, 8))
        kf.H[0, 0] = 1.0
        kf.H[1, 1] = 1.0
        kf.H[2, 2] = 1.0
        kf.H[3, 3] = 1.0
        
        # Set appropriate process and measurement noise
        kf.Q[4:, 4:] *= 10.0  # Process noise (velocity components)
        kf.Q[:4, :4] *= 1.0   # Process noise (position components)
        kf.R *= 10.0          # Measurement noise
        
        # Initialize P with high uncertainty
        kf.P *= 1000.0
        
        return kf

    def update(self, frame, detections):
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Filter detections by confidence
        valid_detections = [det for det in detections if det.confidence > self.conf_threshold]
        
        # Skip update if no detections or no tracks
        if len(valid_detections) == 0:
            # Increment age of all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                # Remove old tracks
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
                    if track_id in self.feature_db:
                        del self.feature_db[track_id]
                    if track_id in self.feature_history:
                        del self.feature_history[track_id]
                    if track_id in self.kalman_filters:
                        del self.kalman_filters[track_id]
            return valid_detections

        # Extract features for detections
        detection_features = []
        for det in valid_detections:
            # Get image crop from detection
            x1, y1, x2, y2 = det.bbox
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            # Skip if crop is invalid
            if crop.size == 0:
                detection_features.append(np.zeros(768))
                continue
            # Extract feature from crop
            feat = self.extract_feature(crop)
            if feat is not None:
                detection_features.append(feat)
            else:
                detection_features.append(np.zeros(768))
        
        # Skip if no valid tracks
        if len(self.tracks) == 0:
            # Create new tracks for all detections
            for i, det in enumerate(valid_detections):
                feat = detection_features[i]
                # Skip invalid features
                if feat is not None and np.sum(feat) != 0:
                    # Create new track using pending track logic
                    self.create_or_update_pending_track(det, feat)
            return valid_detections
        
        # Predict next positions using Kalman filters
        for track_id in list(self.tracks.keys()):
            if track_id in self.kalman_filters:
                self.kalman_filters[track_id].predict()
        
        # Calculate similarity matrix between all track features and detection features
        similarity_matrix = np.zeros((len(valid_detections), len(self.tracks)))
        iou_matrix = np.zeros((len(valid_detections), len(self.tracks)))
        
        # Calculate similarity and IoU between all detections and tracks
        for i, det in enumerate(valid_detections):
            for j, track_id in enumerate(self.tracks):
                # Skip if detection feature is invalid
                if detection_features[i] is None or np.sum(detection_features[i]) == 0:
                    similarity_matrix[i, j] = 0
                else:
                    # Calculate similarity between features
                    similarity_matrix[i, j] = np.dot(self.feature_db[track_id], detection_features[i])
                
                # Calculate IoU between boxes
                iou_matrix[i, j] = self.calculate_iou(det.bbox, self.tracks[track_id]['bbox'])
        
        # Adaptive weighting based on detection confidence and track age
        combined_matrix = np.zeros_like(similarity_matrix)
        for i, det in enumerate(valid_detections):
            for j, track_id in enumerate(list(self.tracks.keys())):
                track_age = self.tracks[track_id]['age']
                det_confidence = getattr(det, 'confidence', 0.6)
                
                # Higher confidence and lower age = more weight on appearance
                appearance_weight = min(0.9, max(0.5, det_confidence - 0.1 * (track_age / self.max_age)))
                combined_matrix[i, j] = appearance_weight * similarity_matrix[i, j] + (1 - appearance_weight) * iou_matrix[i, j]
        
        # Match detections to tracks
        matches = []
        if combined_matrix.size > 0:
            # Use Hungarian algorithm for optimal assignment
            from scipy.optimize import linear_sum_assignment
            det_indices, track_indices = linear_sum_assignment(-combined_matrix)  # Negative for max cost
            
            # Filter matches by threshold
            for det_idx, track_idx in zip(det_indices, track_indices):
                track_id = list(self.tracks.keys())[track_idx]
                # Use a threshold to determine if match is valid
                if combined_matrix[det_idx, track_idx] > 0.2:  # Minimum match quality
                    matches.append((det_idx, track_id))
        
        # Create sets of matched detections and tracks
        matched_det_indices = set([m[0] for m in matches])
        matched_track_ids = set([m[1] for m in matches])
        
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(len(valid_detections)) if i not in matched_det_indices]
        unmatched_tracks = [track_id for track_id in self.tracks if track_id not in matched_track_ids]
        
        # Update matched tracks
        for det_idx, track_id in matches:
            det = valid_detections[det_idx]
            feat = detection_features[det_idx]
            
            # Update track properties
            self.tracks[track_id]['bbox'] = det.bbox
            self.tracks[track_id]['age'] = 0
            # Set track ID on detection
            det.track_id = track_id
            
            # Update feature using the proper history-based method (only if valid feature)
            if feat is not None and np.sum(feat) != 0:
                self.update_track_features(track_id, feat)
            
            # Update Kalman filter with new measurement
            if track_id in self.kalman_filters:
                # Convert box to measurement [x, y, w, h]
                x1, y1, x2, y2 = det.bbox
                w, h = x2-x1, y2-y1
                measurement = np.array([[x1], [y1], [w], [h]])
                self.kalman_filters[track_id].update(measurement)
            else:
                # Initialize Kalman filter if not exists
                self.kalman_filters[track_id] = self.initialize_kalman(det.bbox)
                
            # Update track confidence
            if not hasattr(self, 'track_confidence'):
                self.track_confidence = {}
            self.track_confidence[track_id] = min(1.0, self.track_confidence.get(track_id, 0.8) + 0.1)
        
        # Process unmatched detections - create new tracks or update pending
        for det_idx in unmatched_detections:
            det = valid_detections[det_idx]
            feat = detection_features[det_idx]
            
            # Skip if feature is invalid
            if feat is None or np.sum(feat) == 0:
                det.track_id = None
                continue
            
            # Try to create or update pending track
            if not self.create_or_update_pending_track(det, feat):
                # If not added to pending, track is still untracked
                det.track_id = None
        
        # Update unmatched tracks - use Kalman prediction for occlusions
        for track_id in unmatched_tracks:
            if track_id in self.kalman_filters:
                # Get predicted state
                pred_state = self.kalman_filters[track_id].x
                x1 = float(pred_state[0])
                y1 = float(pred_state[1])
                w = float(pred_state[2])
                h = float(pred_state[3])
                
                # Check if prediction is valid (within frame, reasonable size)
                if (w > 10 and h > 10 and 
                    x1 >= 0 and y1 >= 0 and 
                    x1 + w < frame.shape[1] and y1 + h < frame.shape[0]):
                    
                    # Update track with predicted position during occlusion
                    self.tracks[track_id]['bbox'] = [x1, y1, x1+w, y1+h]
                    
                    # Decay confidence during occlusion
                    if not hasattr(self, 'track_confidence'):
                        self.track_confidence = {}
                    if track_id in self.track_confidence:
                        self.track_confidence[track_id] *= 0.9
                    
                    # Increase age more slowly during predicted occlusion
                    self.tracks[track_id]['age'] += 0.5
                else:
                    # Normal aging for invalid predictions
                    self.tracks[track_id]['age'] += 1
            else:
                # Normal aging for tracks without Kalman
                self.tracks[track_id]['age'] += 1
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['age'] > self.max_age:
                # Save to old_tracks for possible future recovery
                if not hasattr(self, 'old_tracks'):
                    self.old_tracks = {}
                
                # Store for possible re-identification later
                self.old_tracks[track_id] = {
                    'feature': self.feature_db[track_id],
                    'last_seen': self.frame_count
                }
                
                # Remove from current tracks
                del self.tracks[track_id]
                if track_id in self.feature_db:
                    del self.feature_db[track_id]
                if track_id in self.feature_history:
                    del self.feature_history[track_id]
                if track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]
                if track_id in self.track_confidence:
                    del self.track_confidence[track_id]
        
        # Return the updated detections with track IDs
        return valid_detections
                    
    def update_track_features(self, track_id, new_feature):
        """Update track features using history for robustness"""
        # Skip if feature is invalid
        if new_feature is None or np.sum(new_feature) == 0:
            return
            
        # Add to history
        if track_id not in self.feature_history:
            self.feature_history[track_id] = []
        self.feature_history[track_id].append(new_feature)
        
        # Keep only recent features
        if len(self.feature_history[track_id]) > self.max_history:
            self.feature_history[track_id] = self.feature_history[track_id][-self.max_history:]
        
        # Compute robust feature by averaging recent high-quality features
        if len(self.feature_history[track_id]) >= 3:
            # Use median to be more robust against outliers
            self.feature_db[track_id] = np.median(self.feature_history[track_id], axis=0)
        else:
            self.feature_db[track_id] = new_feature
        
        # Normalize
        norm = np.linalg.norm(self.feature_db[track_id])
        if norm > 0:
            self.feature_db[track_id] = self.feature_db[track_id] / norm

    def create_or_update_pending_track(self, det, feat):
        """Create a new pending track or update existing one"""
        # Skip if feature is invalid
        if feat is None or np.sum(feat) == 0:
            return False
            
        # Create a track signature (can be center position or feature vector)
        track_sig = self.get_track_signature(det, feat)
        
        if track_sig in self.pending_tracks:
            self.pending_tracks[track_sig]['count'] += 1
            self.pending_tracks[track_sig]['feature'] = 0.7 * self.pending_tracks[track_sig]['feature'] + 0.3 * feat
            
            # Confirm track if it appears enough times
            if self.pending_tracks[track_sig]['count'] >= self.confirmed_threshold:
                new_id = max(1, self.next_id)
                self.tracks[new_id] = {
                    'bbox': det.bbox,
                    'feature': self.pending_tracks[track_sig]['feature'],
                    'age': 0
                }
                self.feature_db[new_id] = self.pending_tracks[track_sig]['feature']
                self.track_confidence[new_id] = 1.0
                self.kalman_filters[new_id] = self.initialize_kalman(det.bbox)
                det.track_id = new_id
                self.next_id = new_id + 1
                
                # Remove from pending
                del self.pending_tracks[track_sig]
                return True
        else:
            # Create new pending track
            self.pending_tracks[track_sig] = {
                'count': 1,
                'feature': feat,
                'bbox': det.bbox
            }
        
        return False

    def get_track_signature(self, det, feat):
        """Generate a unique signature for a detection based on position and features"""
        # Use center position as a simple signature
        x1, y1, x2, y2 = det.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Quantize to reduce small position differences
        center_x_q = int(center_x / 10) * 10  
        center_y_q = int(center_y / 10) * 10
        return f"{center_x_q}_{center_y_q}"

class DetectionTracker:
    """Main detection and tracking controller"""
    
    def __init__(self, device='cpu', pose_device='cpu', use_transreid=True, 
                 transreid_model='weights/transreid_vitbase.pth', tracking_iou=0.5, tracking_age=30):
        self.device = device
        self.pose_device = pose_device
        
        # Initialize YOLO models
        self.model_det = YOLO('weights/yolo11x.pt').to(device)
        self.model_pose = YOLO('weights/yolo11x-pose.pt').to(pose_device)
        
        # Initialize trackers
        self.yolo_tracker = YOLOTracker(self.model_det, device=device)
        
        self.transreid_tracker = None
        if use_transreid:
            print("Using TransReID for robust re-identification and tracking")
            self.transreid_tracker = TransReIDTracker(
                device=device, 
                reid_model_path=transreid_model,
                iou_threshold=tracking_iou,
                max_age=tracking_age
            )
        else:
            print("Using YOLO's built-in tracking (not recommended)")
    
    def detect_and_track(self, frame):
        """Detect people and track them across frames"""
        
        try:
            if self.transreid_tracker:
                # For TransReID: First detect without tracking, then use TransReID for tracking
                results = self.model_det(frame, conf=0.45, classes=[0], device=self.device, verbose=False)
                
                detections = []
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        detections.append(Detection(
                            bbox=box,
                            confidence=conf,
                            track_id=None  # No track ID yet
                        ))
                
                # Use TransReID for tracking
                tracked_detections = self.transreid_tracker.update(frame, detections)
                
                # Ensure we always return a list, never None
                if tracked_detections is None:
                    print("WARNING: TransReIDTracker.update returned None, returning empty list")
                    tracked_detections = []
                    
            else:
                # Use YOLO's built-in tracking (detection + tracking in one step)
                tracked_detections = self.yolo_tracker.update(frame)
                
                # Ensure we always return a list, never None
                if tracked_detections is None:
                    print("WARNING: YOLOTracker.update returned None, returning empty list")
                    tracked_detections = []
            
            return tracked_detections
            
        except Exception as e:
            print(f"ERROR in detect_and_track: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list in case of any exception
            return []
