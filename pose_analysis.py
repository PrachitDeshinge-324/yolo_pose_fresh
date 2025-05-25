"""
Pose Analysis Module

Handles pose detection, keypoint processing, and RAW keypoint storage.
Feature extraction is delayed until after ID merging.
"""

import numpy as np
from collections import defaultdict
from utils.pose_detector import PoseDetector

class PoseAnalyzer:
    """Pose analysis with delayed feature extraction"""
    
    def __init__(self, model_pose, pose_device='cpu', history_length=5):
        self.pose_detector = PoseDetector(model_pose, device=pose_device)
        
        # Keypoints history for temporal smoothing
        self.keypoints_history = defaultdict(list)
        self.history_length = history_length
        
        # 🔥 Store RAW keypoints only - no feature extraction yet
        self.raw_keypoints_storage = defaultdict(list)  # track_id -> [(frame_idx, keypoints), ...]
        
        # Person identification (optional) - will be set later after feature extraction
        self.person_identifier = None
        
        # Feature extraction components - will be created later with merged data
        self.gait_analyzer = None
        self.gait_validator = None
    
    def process_pose(self, frame, detection, frame_count):
        """Process pose for a single person detection - STORE RAW KEYPOINTS ONLY"""
        track_id = detection.track_id
        
        if track_id <= 0:  # Skip invalid track IDs
            return None
            
        # Extract person crop with buffer
        x1, y1, x2, y2 = map(int, detection.bbox)
        buffer_x = int((x2 - x1) * 0.05)
        buffer_y = int((y2 - y1) * 0.05)
        
        x1_buf = max(0, x1 - buffer_x)
        y1_buf = max(0, y1 - buffer_y)
        x2_buf = min(frame.shape[1], x2 + buffer_x)
        y2_buf = min(frame.shape[0], y2 + buffer_y)
        
        person_crop = frame[y1_buf:y2_buf, x1_buf:x2_buf]
        
        if person_crop.size == 0:
            return None
        
        # Detect poses in the crop
        keypoints_list = self.pose_detector.detect(person_crop)
        
        processed_keypoints = []
        for keypoints in keypoints_list:
            # Update keypoints history for smoothing
            if len(self.keypoints_history[track_id]) >= self.history_length:
                self.keypoints_history[track_id].pop(0)
            self.keypoints_history[track_id].append(keypoints.copy())
            
            # Apply temporal smoothing to keypoints
            smoothed_keypoints = self.pose_detector.smooth_keypoints(
                self.keypoints_history[track_id][:-1], keypoints)
            
            # 🔥 Store RAW smoothed keypoints - NO feature extraction yet
            self.raw_keypoints_storage[track_id].append((frame_count, smoothed_keypoints))
            
            # Transform keypoints back to original frame coordinates for visualization
            transformed_keypoints = self._transform_keypoints_to_original(
                smoothed_keypoints, (x1_buf, y1_buf))
            
            processed_keypoints.append({
                'keypoints': transformed_keypoints,
                'crop_offset': (x1_buf, y1_buf),
                'identity': None,  # No identity matching yet - will be done after feature extraction
                'confidence': 0.0
            })
        
        return processed_keypoints
    
    def _transform_keypoints_to_original(self, keypoints, crop_offset):
        """Transform keypoints from crop coordinates back to original frame coordinates"""
        if keypoints is None:
            return None
        
        x_offset, y_offset = crop_offset
        transformed = keypoints.copy()
        
        # Add crop offset to each keypoint
        for i in range(len(transformed)):
            if not np.isnan(transformed[i]).any():
                transformed[i][0] += x_offset  # x coordinate
                transformed[i][1] += y_offset  # y coordinate
        
        return transformed
    
    def get_raw_keypoints(self):
        """Get all stored raw keypoints"""
        return dict(self.raw_keypoints_storage)
    
    def get_track_history(self):
        """Get track history for summary"""
        return dict(self.raw_keypoints_storage)
    
    def get_track_summary(self):
        """Get summary of collected tracks"""
        summary = {}
        for track_id, keypoints_list in self.raw_keypoints_storage.items():
            summary[track_id] = {
                'total_frames': len(keypoints_list),
                'first_frame': min(frame_idx for frame_idx, _ in keypoints_list) if keypoints_list else 0,
                'last_frame': max(frame_idx for frame_idx, _ in keypoints_list) if keypoints_list else 0
            }
        return summary
    
    def clear_storage(self):
        """Clear all stored data"""
        self.raw_keypoints_storage.clear()
        self.keypoints_history.clear()
    
    # Methods that will be available after feature processing
    def create_feature_analyzer_with_merged_data(self, merged_keypoints):
        """Create and populate feature analyzer with merged keypoint data"""
        from utils.skeleton_gait import NormalizedGaitFeatureExtractor
        from utils.gait_validator import GaitValidator
        
        gait_analyzer = NormalizedGaitFeatureExtractor()
        
        # Add all merged keypoints to the analyzer
        for track_id, keypoints_list in merged_keypoints.items():
            for frame_idx, keypoints in keypoints_list:
                gait_analyzer.update_track(track_id, keypoints, frame_idx)
        
        return gait_analyzer
    
    # Legacy compatibility methods (will work after feature processing is done)
    def set_gait_analyzer(self, gait_analyzer):
        """Set gait analyzer after feature processing"""
        from utils.gait_validator import GaitValidator
        self.gait_analyzer = gait_analyzer
        self.gait_validator = GaitValidator()
    
    def get_gait_features(self, track_id):
        """Get extracted gait features for a track (only works after feature processing)"""
        if hasattr(self, 'gait_analyzer') and self.gait_analyzer:
            return self.gait_analyzer.get_features(track_id)
        else:
            print("Warning: Features not yet computed. Run feature processing first.")
            return None
    
    def get_feature_vector(self, track_id):
        """Get feature vector for person identification (only works after feature processing)"""
        if hasattr(self, 'gait_analyzer') and self.gait_analyzer:
            return self.gait_analyzer.get_feature_vector(track_id)
        else:
            print("Warning: Features not yet computed. Run feature processing first.")
            return None
    
    def export_features_csv(self, csv_path):
        """Export all gait features to CSV (only works after feature processing)"""
        if hasattr(self, 'gait_analyzer') and self.gait_analyzer:
            return self.gait_analyzer.export_features_csv(csv_path)
        else:
            print("Warning: Features not yet computed. Run feature processing first.")
            return False
    
    def validate_gait_sequence(self, track_id):
        """Validate if gait sequence is complete and reliable (only works after feature processing)"""
        if hasattr(self, 'gait_analyzer') and self.gait_analyzer and hasattr(self, 'gait_validator'):
            features = self.get_gait_features(track_id)
            if features is None:
                return False
            return self.gait_validator.validate_sequence(features)
        else:
            print("Warning: Features not yet computed. Run feature processing first.")
            return False
    
    def calculate_view_invariant_features(self, track_id):
        """Calculate view-invariant gait features (only works after feature processing)"""
        if hasattr(self, 'gait_analyzer') and self.gait_analyzer:
            return self.gait_analyzer.calculate_view_invariant_features(track_id)
        else:
            print("Warning: Features not yet computed. Run feature processing first.")
            return None
    
    def set_person_identifier(self, identifier):
        """Set person identifier for gait-based recognition"""
        self.person_identifier = identifier
    
    def get_normalized_keypoints(self, track_id):
        """Get normalized keypoints for a track (only works after feature processing)"""
        if hasattr(self, 'gait_analyzer') and self.gait_analyzer and track_id in self.gait_analyzer.track_history:
            history = self.gait_analyzer.track_history[track_id]
            normalized_points = []
            for frame_idx, keypoints in history:
                normalized = self.gait_analyzer.normalize_keypoints(keypoints)
                normalized_points.append((frame_idx, normalized))
            return normalized_points
        else:
            print("Warning: Features not yet computed. Run feature processing first.")
            return None
    
    def get_gait_cycle_info(self, track_id):
        """Get gait cycle information for a track (only works after feature processing)"""
        features = self.get_gait_features(track_id)
        if features is None:
            return None
            
        # Extract gait cycle specific metrics
        cycle_info = {
            'step_frequency': features.get('step_frequency', 0),
            'stride_length': features.get('avg_stride_length', 0),
            'gait_symmetry': features.get('gait_symmetry_ratio', 0),
            'stance_phase_ratio': features.get('stance_phase_ratio', 0)
        }
        
        return cycle_info
    
    def get_postural_features(self, track_id):
        """Get postural and body proportion features (only works after feature processing)"""
        features = self.get_gait_features(track_id)
        if features is None:
            return None
            
        postural_info = {
            'body_height_ratio': features.get('avg_body_height', 0),
            'torso_leg_ratio': features.get('avg_torso_to_leg_ratio', 0),
            'shoulder_width': features.get('avg_shoulder_width', 0),
            'hip_width': features.get('avg_hip_width', 0)
        }
        
        return postural_info
    
    def calculate_frame_features(self, track_id, frame_idx):
        """Calculate features for current frame of a specific track"""
        if not hasattr(self, 'gait_analyzer') or self.gait_analyzer is None:
            # Create gait analyzer if not exists
            from utils.skeleton_gait import NormalizedGaitFeatureExtractor
            self.gait_analyzer = NormalizedGaitFeatureExtractor()
        
        # Get current keypoints for this track
        if track_id in self.raw_keypoints_storage:
            # Find keypoints for this frame
            keypoints_for_frame = None
            for stored_frame_idx, keypoints in self.raw_keypoints_storage[track_id]:
                if stored_frame_idx == frame_idx:
                    keypoints_for_frame = keypoints
                    break
            
            if keypoints_for_frame is not None:
                # Add to gait analyzer
                self.gait_analyzer.update_track(track_id, keypoints_for_frame, frame_idx)
                
                # 🔥 Calculate frame-level features from keypoints directly
                frame_features = self._calculate_frame_level_features(keypoints_for_frame, track_id, frame_idx)
                
                return frame_features
        
        return None

    def _calculate_frame_level_features(self, keypoints, track_id, frame_idx):
        """Calculate features for a single frame's keypoints"""
        try:
            # Initialize features dictionary
            features = {}
            
            # Basic keypoint validation
            if keypoints is None or len(keypoints) == 0:
                return {}
            
            # Convert to numpy array if needed
            if not isinstance(keypoints, np.ndarray):
                keypoints = np.array(keypoints)
            
            # Check if we have enough keypoints (assuming MediaPipe 33 keypoints or similar)
            if len(keypoints) < 10:
                return {}
            
            # 🔥 Calculate basic geometric features for this frame
            
            # 1. Body height (head to foot distance)
            if len(keypoints) >= 33:  # MediaPipe format
                # Head keypoints (approximate)
                head_y = keypoints[0][1] if len(keypoints) > 0 else 0
                # Foot keypoints (approximate) 
                left_foot_y = keypoints[31][1] if len(keypoints) > 31 else keypoints[-1][1]
                right_foot_y = keypoints[32][1] if len(keypoints) > 32 else keypoints[-1][1]
                foot_y = max(left_foot_y, right_foot_y)
                
                features['frame_body_height'] = abs(foot_y - head_y) if foot_y != head_y else 0
            
            # 2. Center of mass (approximate)
            valid_keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
            if len(valid_keypoints) > 0:
                features['frame_center_x'] = np.mean(valid_keypoints[:, 0])
                features['frame_center_y'] = np.mean(valid_keypoints[:, 1])
            
            # 3. Keypoint spread (body width approximation)
            if len(valid_keypoints) > 0:
                features['frame_width'] = np.max(valid_keypoints[:, 0]) - np.min(valid_keypoints[:, 0])
                features['frame_height'] = np.max(valid_keypoints[:, 1]) - np.min(valid_keypoints[:, 1])
            
            # 4. Joint angles (if we have enough keypoints)
            if len(keypoints) >= 33:  # MediaPipe format
                # Calculate some basic joint angles
                features.update(self._calculate_basic_joint_angles(keypoints))
            
            # 5. Movement features (if we have history)
            if hasattr(self, 'gait_analyzer') and self.gait_analyzer:
                movement_features = self._calculate_movement_features(track_id, frame_idx, keypoints)
                features.update(movement_features)
            
            # 6. Add frame metadata
            features['track_id'] = track_id
            features['frame_idx'] = frame_idx
            features['num_valid_keypoints'] = len(valid_keypoints)
            
            return features
            
        except Exception as e:
            print(f"Error calculating frame features for track {track_id}, frame {frame_idx}: {e}")
            return {}

    def _calculate_basic_joint_angles(self, keypoints):
        """Calculate basic joint angles from keypoints"""
        angles = {}
        
        try:
            # Assuming MediaPipe pose landmarks
            # Left knee angle (hip-knee-ankle)
            if len(keypoints) > 27:
                left_hip = keypoints[23]
                left_knee = keypoints[25] 
                left_ankle = keypoints[27]
                
                if not (np.isnan(left_hip).any() or np.isnan(left_knee).any() or np.isnan(left_ankle).any()):
                    angles['left_knee_angle'] = self._calculate_angle(left_hip, left_knee, left_ankle)
            
            # Right knee angle
            if len(keypoints) > 28:
                right_hip = keypoints[24]
                right_knee = keypoints[26]
                right_ankle = keypoints[28]
                
                if not (np.isnan(right_hip).any() or np.isnan(right_knee).any() or np.isnan(right_ankle).any()):
                    angles['right_knee_angle'] = self._calculate_angle(right_hip, right_knee, right_ankle)
            
            # Add more joint angles as needed...
            
        except Exception as e:
            print(f"Error calculating joint angles: {e}")
        
        return angles

    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Vectors
            v1 = point1 - point2
            v2 = point3 - point2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return 0.0

    def _calculate_movement_features(self, track_id, frame_idx, current_keypoints):
        """Calculate movement features based on keypoint history"""
        movement_features = {}
        
        try:
            # Get previous keypoints for this track
            if track_id in self.raw_keypoints_storage:
                # Find previous frame keypoints
                previous_keypoints = None
                for stored_frame_idx, keypoints in self.raw_keypoints_storage[track_id]:
                    if stored_frame_idx == frame_idx - 1:
                        previous_keypoints = keypoints
                        break
                
                if previous_keypoints is not None:
                    # Calculate movement between frames
                    if len(current_keypoints) == len(previous_keypoints):
                        # Calculate displacement
                        displacement = current_keypoints - previous_keypoints
                        
                        # Remove invalid displacements
                        valid_displacement = displacement[~np.isnan(displacement).any(axis=1)]
                        
                        if len(valid_displacement) > 0:
                            movement_features['frame_avg_displacement'] = np.mean(np.linalg.norm(valid_displacement, axis=1))
                            movement_features['frame_max_displacement'] = np.max(np.linalg.norm(valid_displacement, axis=1))
                            movement_features['frame_displacement_std'] = np.std(np.linalg.norm(valid_displacement, axis=1))
        
        except Exception as e:
            print(f"Error calculating movement features: {e}")
        
        return movement_features