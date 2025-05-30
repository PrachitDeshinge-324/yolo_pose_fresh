"""
Enhanced Gait Recognition with Identity Conflict Resolution
"""
import argparse
import os
import torch
import numpy as np
import cv2
import json
import joblib
from collections import defaultdict, deque, Counter
from sklearn.preprocessing import StandardScaler

# Import existing components
from detection_tracking import DetectionTracker
from pose_analysis import PoseAnalyzer
from utils.visualization import Visualizer
from train_lstm_gait import BidirectionalLSTM, get_best_device

class EnhancedGaitInferenceWithConflictResolution:
    """Enhanced inference with identity conflict resolution"""
    
    def __init__(self, model_path, sequence_length=20):
        self.device = get_best_device()
        self.sequence_length = sequence_length
        
        # Load model and configuration
        self.load_model_and_config(model_path)
        
        # Initialize gait feature extractor
        from utils.skeleton_gait import NormalizedGaitFeatureExtractor
        self.gait_extractor = NormalizedGaitFeatureExtractor()
        
        # Enhanced tracking for stability
        self.track_features = defaultdict(lambda: deque(maxlen=sequence_length))
        self.track_predictions = defaultdict(list)
        self.track_confidences = defaultdict(list)
        self.track_all_probabilities = defaultdict(list)
        self.track_states = defaultdict(str)
        self.stable_identities = {}
        
        # Frame-based displacement tracking
        self.track_positions = defaultdict(list)
        self.track_displacements = defaultdict(list)
        
        # Identity conflict resolution
        self.identity_assignments = {}  # person_name -> track_id
        self.alternative_identities = {}  # track_id -> list of alternative names
        self.conflict_resolution_enabled = True
        
        # More conservative thresholds for better discrimination
        self.min_predictions_for_stability = 10
        self.confidence_threshold = 0.4  # Higher threshold
        self.consistency_threshold = 0.8  # Higher consistency required
        self.entropy_threshold = 1.4  # Lower entropy for more certainty
        self.uniqueness_threshold = 0.1  # Minimum difference between top predictions
        
        # Debug statistics
        self.debug_stats = {
            'total_predictions': 0,
            'feature_extraction_errors': 0,
            'prediction_errors': 0,
            'class_distribution': Counter(),
            'feature_counts': [],
            'identity_conflicts': 0,
            'resolved_conflicts': 0
        }
        
        print(f"✅ Enhanced Gait Inference with Conflict Resolution Initialized")
        print(f"   Model: {len(self.class_names)} classes")
        print(f"   Features: {self.input_size}")
        print(f"   Device: {self.device}")
        print(f"   Enhanced thresholds: conf={self.confidence_threshold}, consistency={self.consistency_threshold}")
        print(f"   Conflict resolution: {'enabled' if self.conflict_resolution_enabled else 'disabled'}")

    def load_model_and_config(self, model_path):
        """Load model with proper configuration"""
        print(f"Loading model from {model_path}...")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.sequence_length = config['sequence_length']
                hidden_size = config['hidden_size']
                num_layers = config['num_layers']
                dropout = config['dropout']
            else:
                hidden_size = 64
                num_layers = 2
                dropout = 0.2
            
            # Get model dimensions
            fc3_weight = checkpoint['model_state_dict']['fc3.weight']
            input_norm_weight = checkpoint['model_state_dict']['input_norm.weight']
            
            num_classes = fc3_weight.shape[0]
            self.input_size = input_norm_weight.shape[0]
            
            print(f"🔍 Model Configuration:")
            print(f"   Input size: {self.input_size}")
            print(f"   Classes: {num_classes}")
            print(f"   Hidden size: {hidden_size}")
            print(f"   Sequence length: {self.sequence_length}")
            
            # Create and load model
            self.model = BidirectionalLSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load label encoder
            if 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
                self.class_names = [f"Person_{track_id}" for track_id in self.label_encoder.classes_]
                print(f"   Class mapping: {dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))}")
            else:
                self.class_names = [f"Person_{i+1}" for i in range(num_classes)]
                self.label_encoder = None
            
            # Load feature scaler
            model_dir = os.path.dirname(model_path)
            scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Loaded feature scaler")
                if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                    print(f"   Scaler stats: mean range [{self.scaler.mean_.min():.3f}, {self.scaler.mean_.max():.3f}]")
                    print(f"   Scaler stats: scale range [{self.scaler.scale_.min():.3f}, {self.scaler.scale_.max():.3f}]")
            else:
                print(f"⚠️ No feature scaler found")
                self.scaler = None
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def safe_extract_keypoint_features(self, keypoints):
        """Safely extract keypoint features with proper error handling"""
        features = {}
        
        try:
            if keypoints is None or len(keypoints) == 0:
                return features
            
            # Handle different keypoint formats
            if isinstance(keypoints, (list, tuple)):
                if len(keypoints) == 0:
                    return features
                
                # Check if first element is a coordinate pair/triple
                first_kp = keypoints[0]
                if isinstance(first_kp, (list, tuple, np.ndarray)):
                    # Format: [[x1, y1], [x2, y2], ...] or [[x1, y1, conf1], [x2, y2, conf2], ...]
                    keypoints_array = np.array(keypoints)
                    
                    if keypoints_array.ndim == 2 and keypoints_array.shape[1] >= 2:
                        # Ensure we have at least 17 keypoints for COCO format
                        if keypoints_array.shape[0] >= 17:
                            # Add confidence column if missing
                            if keypoints_array.shape[1] == 2:
                                conf_col = np.ones((keypoints_array.shape[0], 1))
                                keypoints_array = np.hstack([keypoints_array, conf_col])
                            
                            # Now extract features safely
                            features.update(self.extract_coco_features(keypoints_array))
                else:
                    # Flat format: [x1, y1, conf1, x2, y2, conf2, ...]
                    if len(keypoints) >= 51:  # 17 * 3 = 51
                        keypoints_reshaped = np.array(keypoints[:51]).reshape(17, 3)
                        features.update(self.extract_coco_features(keypoints_reshaped))
            
            elif isinstance(keypoints, np.ndarray):
                if keypoints.ndim == 2 and keypoints.shape[0] >= 17 and keypoints.shape[1] >= 2:
                    # Add confidence if missing
                    if keypoints.shape[1] == 2:
                        conf_col = np.ones((keypoints.shape[0], 1))
                        keypoints = np.hstack([keypoints, conf_col])
                    features.update(self.extract_coco_features(keypoints))
                elif keypoints.ndim == 1 and len(keypoints) >= 51:
                    keypoints_reshaped = keypoints[:51].reshape(17, 3)
                    features.update(self.extract_coco_features(keypoints_reshaped))
            
            return features
            
        except Exception as e:
            return {}

    def extract_coco_features(self, keypoints):
        """Extract features from COCO format keypoints [17, 3]"""
        features = {}
        
        try:
            # Key indices
            LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
            LEFT_ELBOW, RIGHT_ELBOW = 7, 8
            LEFT_WRIST, RIGHT_WRIST = 9, 10
            LEFT_HIP, RIGHT_HIP = 11, 12
            LEFT_KNEE, RIGHT_KNEE = 13, 14
            LEFT_ANKLE, RIGHT_ANKLE = 15, 16
            NOSE = 0
            
            # Calculate knee angles
            if (keypoints[LEFT_HIP, 2] > 0.3 and keypoints[LEFT_KNEE, 2] > 0.3 and keypoints[LEFT_ANKLE, 2] > 0.3):
                features['left_knee_angle'] = self.calculate_angle(
                    keypoints[LEFT_HIP, :2], keypoints[LEFT_KNEE, :2], keypoints[LEFT_ANKLE, :2]
                )
            else:
                features['left_knee_angle'] = 90.0
            
            if (keypoints[RIGHT_HIP, 2] > 0.3 and keypoints[RIGHT_KNEE, 2] > 0.3 and keypoints[RIGHT_ANKLE, 2] > 0.3):
                features['right_knee_angle'] = self.calculate_angle(
                    keypoints[RIGHT_HIP, :2], keypoints[RIGHT_KNEE, :2], keypoints[RIGHT_ANKLE, :2]
                )
            else:
                features['right_knee_angle'] = 90.0
            
            # Calculate body height
            valid_top = keypoints[NOSE, 2] > 0.3
            valid_left_ankle = keypoints[LEFT_ANKLE, 2] > 0.3
            valid_right_ankle = keypoints[RIGHT_ANKLE, 2] > 0.3
            
            if valid_top and (valid_left_ankle or valid_right_ankle):
                if valid_left_ankle:
                    features['frame_body_height'] = abs(keypoints[NOSE, 1] - keypoints[LEFT_ANKLE, 1])
                else:
                    features['frame_body_height'] = abs(keypoints[NOSE, 1] - keypoints[RIGHT_ANKLE, 1])
            else:
                features['frame_body_height'] = 200.0
            
            # Count valid keypoints
            features['num_valid_keypoints'] = int(np.sum(keypoints[:, 2] > 0.3))
            
            return features
            
        except Exception as e:
            return {}

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            return float(angle)
        except:
            return 90.0

    def calculate_displacement_features(self, track_id, bbox):
        """Calculate movement-based features"""
        features = {}
        
        try:
            # Get center of bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Store position
            self.track_positions[track_id].append([center_x, center_y])
            
            # Keep only recent positions
            if len(self.track_positions[track_id]) > 10:
                self.track_positions[track_id] = self.track_positions[track_id][-10:]
            
            # Calculate displacements
            if len(self.track_positions[track_id]) >= 2:
                positions = np.array(self.track_positions[track_id])
                displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                
                self.track_displacements[track_id] = displacements.tolist()
                
                # Calculate displacement features
                features['frame_avg_displacement'] = float(np.mean(displacements))
                features['frame_max_displacement'] = float(np.max(displacements))
                features['frame_displacement_std'] = float(np.std(displacements))
            else:
                features['frame_avg_displacement'] = 0.0
                features['frame_max_displacement'] = 0.0
                features['frame_displacement_std'] = 0.0
            
            return features
            
        except Exception as e:
            return {
                'frame_avg_displacement': 0.0,
                'frame_max_displacement': 0.0,
                'frame_displacement_std': 0.0
            }

    def build_complete_feature_vector(self, track_id, basic_features, keypoints, bbox):
        """Build complete feature vector with all components"""
        
        # Initialize features dictionary
        all_features = {}
        
        # 1. Add basic frame features
        if basic_features:
            all_features.update(basic_features)
        
        # 2. Add keypoint-based features
        keypoint_features = self.safe_extract_keypoint_features(keypoints)
        all_features.update(keypoint_features)
        
        # 3. Add displacement features
        displacement_features = self.calculate_displacement_features(track_id, bbox)
        all_features.update(displacement_features)
        
        # 4. Update gait extractor and get gait features
        if keypoints is not None:
            try:
                # Convert keypoints to format expected by gait extractor
                if isinstance(keypoints, (list, tuple, np.ndarray)):
                    keypoints_array = np.array(keypoints)
                    if keypoints_array.ndim == 2 and keypoints_array.shape[0] >= 17:
                        self.gait_extractor.update_track(track_id, keypoints_array, len(self.track_features[track_id]))
                        gait_features = self.gait_extractor.get_features(track_id)
                        if gait_features:
                            all_features.update(gait_features)
            except Exception as e:
                pass  # Silently continue if gait extraction fails
        
        # 5. Build final feature vector
        feature_vector = self.map_to_training_features(all_features)
        
        return feature_vector

    def map_to_training_features(self, features):
        """Map features to exact training format"""
        
        # Expected 41 features in training order
        expected_features = [
            'frame_center_x', 'frame_center_y', 'frame_width', 'frame_height',
            'frame_body_height', 'num_valid_keypoints', 'left_knee_angle', 'right_knee_angle',
            'frame_avg_displacement', 'frame_max_displacement', 'frame_displacement_std',
            'norm_norm_ratio_neck_to_right_shoulder_to_shoulder',
            'norm_norm_ratio_neck_to_left_shoulder_to_shoulder',
            'norm_norm_ratio_right_shoulder_to_right_elbow_to_shoulder',
            'norm_norm_ratio_right_elbow_to_right_wrist_to_shoulder',
            'norm_norm_ratio_left_shoulder_to_left_elbow_to_shoulder',
            'norm_norm_ratio_left_elbow_to_left_wrist_to_shoulder',
            'norm_norm_ratio_right_hip_to_right_knee_to_shoulder',
            'norm_norm_ratio_right_knee_to_right_ankle_to_shoulder',
            'norm_norm_ratio_left_hip_to_left_knee_to_shoulder',
            'norm_norm_ratio_left_knee_to_left_ankle_to_shoulder',
            'norm_norm_arm_symmetry_upper',
            'norm_norm_arm_symmetry_lower',
            'norm_norm_leg_symmetry_upper',
            'norm_norm_leg_symmetry_lower',
            'norm_norm_avg_right_elbow_angle',
            'norm_norm_std_right_elbow_angle',
            'norm_norm_range_right_elbow_angle',
            'norm_norm_avg_left_elbow_angle',
            'norm_norm_std_left_elbow_angle',
            'norm_norm_range_left_elbow_angle',
            'norm_norm_avg_right_knee_angle',
            'norm_norm_std_right_knee_angle',
            'norm_norm_range_right_knee_angle',
            'norm_norm_avg_left_knee_angle',
            'norm_norm_std_left_knee_angle',
            'norm_norm_range_left_knee_angle',
            'norm_norm_movement_mean',
            'norm_norm_movement_std',
            'norm_norm_movement_max',
            'norm_norm_movement_rhythm'
        ]
        
        # Feature mapping for gait features
        gait_mapping = {
            'norm_ratio_neck_to_right_shoulder_to_shoulder': 'norm_norm_ratio_neck_to_right_shoulder_to_shoulder',
            'norm_ratio_neck_to_left_shoulder_to_shoulder': 'norm_norm_ratio_neck_to_left_shoulder_to_shoulder',
            'norm_ratio_right_shoulder_to_right_elbow_to_shoulder': 'norm_norm_ratio_right_shoulder_to_right_elbow_to_shoulder',
            'norm_ratio_right_elbow_to_right_wrist_to_shoulder': 'norm_norm_ratio_right_elbow_to_right_wrist_to_shoulder',
            'norm_ratio_left_shoulder_to_left_elbow_to_shoulder': 'norm_norm_ratio_left_shoulder_to_left_elbow_to_shoulder',
            'norm_ratio_left_elbow_to_left_wrist_to_shoulder': 'norm_norm_ratio_left_elbow_to_left_wrist_to_shoulder',
            'norm_ratio_right_hip_to_right_knee_to_shoulder': 'norm_norm_ratio_right_hip_to_right_knee_to_shoulder',
            'norm_ratio_right_knee_to_right_ankle_to_shoulder': 'norm_norm_ratio_right_knee_to_right_ankle_to_shoulder',
            'norm_ratio_left_hip_to_left_knee_to_shoulder': 'norm_norm_ratio_left_hip_to_left_knee_to_shoulder',
            'norm_ratio_left_knee_to_left_ankle_to_shoulder': 'norm_norm_ratio_left_knee_to_left_ankle_to_shoulder',
            'norm_arm_symmetry_upper': 'norm_norm_arm_symmetry_upper',
            'norm_arm_symmetry_lower': 'norm_norm_arm_symmetry_lower',
            'norm_leg_symmetry_upper': 'norm_norm_leg_symmetry_upper',
            'norm_leg_symmetry_lower': 'norm_norm_leg_symmetry_lower',
            'norm_avg_right_elbow_angle': 'norm_norm_avg_right_elbow_angle',
            'norm_std_right_elbow_angle': 'norm_norm_std_right_elbow_angle',
            'norm_range_right_elbow_angle': 'norm_norm_range_right_elbow_angle',
            'norm_avg_left_elbow_angle': 'norm_norm_avg_left_elbow_angle',
            'norm_std_left_elbow_angle': 'norm_norm_std_left_elbow_angle',
            'norm_range_left_elbow_angle': 'norm_norm_range_left_elbow_angle',
            'norm_avg_right_knee_angle': 'norm_norm_avg_right_knee_angle',
            'norm_std_right_knee_angle': 'norm_norm_std_right_knee_angle',
            'norm_range_right_knee_angle': 'norm_norm_range_right_knee_angle',
            'norm_avg_left_knee_angle': 'norm_norm_avg_left_knee_angle',
            'norm_std_left_knee_angle': 'norm_norm_std_left_knee_angle',
            'norm_range_left_knee_angle': 'norm_norm_range_left_knee_angle',
            'norm_movement_mean': 'norm_norm_movement_mean',
            'norm_movement_std': 'norm_norm_movement_std',
            'norm_movement_max': 'norm_norm_movement_max',
            'norm_movement_rhythm': 'norm_norm_movement_rhythm'
        }
        
        # Build feature vector
        feature_vector = []
        missing_count = 0
        
        for expected_feature in expected_features:
            value = None
            
            # Try direct lookup
            if expected_feature in features:
                value = features[expected_feature]
            else:
                # Try gait mapping
                for source, target in gait_mapping.items():
                    if target == expected_feature and source in features:
                        value = features[source]
                        break
            
            if value is not None and not (np.isnan(value) or np.isinf(value)):
                feature_vector.append(float(value))
            else:
                # Use reasonable defaults based on feature type
                if 'angle' in expected_feature:
                    feature_vector.append(90.0)  # Neutral angle
                elif 'displacement' in expected_feature:
                    feature_vector.append(1.0)  # Small movement
                elif 'height' in expected_feature:
                    feature_vector.append(200.0)  # Default height
                elif 'center' in expected_feature:
                    feature_vector.append(320.0)  # Center of typical frame
                elif 'width' in expected_feature or 'height' in expected_feature:
                    feature_vector.append(100.0)  # Default size
                else:
                    feature_vector.append(0.5)  # Generic default
                missing_count += 1
        
        # Track feature completeness
        self.debug_stats['feature_counts'].append(41 - missing_count)
        
        return np.array(feature_vector, dtype=np.float32)

    def update_track(self, track_id, basic_features, keypoints=None, frame_idx=None, bbox=None):
        """Update track with comprehensive feature extraction"""
        
        try:
            # Build complete feature vector
            feature_vector = self.build_complete_feature_vector(
                track_id, basic_features, keypoints, bbox
            )
            
            if feature_vector is not None and len(feature_vector) == 41:
                # Add to buffer
                self.track_features[track_id].append(feature_vector)
                
                # Update state
                if len(self.track_features[track_id]) < self.sequence_length:
                    self.track_states[track_id] = 'collecting'
                else:
                    self.track_states[track_id] = 'ready'
                    
                    # Make prediction
                    self.make_prediction(track_id)
            else:
                self.debug_stats['feature_extraction_errors'] += 1
                
        except Exception as e:
            self.debug_stats['feature_extraction_errors'] += 1

    def make_prediction(self, track_id):
        """Make prediction with enhanced discrimination and conflict resolution"""
        if len(self.track_features[track_id]) < self.sequence_length:
            return
        
        try:
            # Get feature sequence
            feature_sequence = np.array(list(self.track_features[track_id]))
            
            # Apply scaling if available
            if self.scaler is not None and hasattr(self.scaler, 'mean_'):
                original_shape = feature_sequence.shape
                feature_sequence_flat = feature_sequence.reshape(-1, feature_sequence.shape[-1])
                feature_sequence_scaled = self.scaler.transform(feature_sequence_flat)
                feature_sequence = feature_sequence_scaled.reshape(original_shape)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output, attention_weights = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                probabilities_np = probabilities.cpu().numpy()[0]
                
                # Enhanced prediction analysis
                sorted_probs = sorted(enumerate(probabilities_np), key=lambda x: x[1], reverse=True)
                top_class_idx, top_confidence = sorted_probs[0]
                second_class_idx, second_confidence = sorted_probs[1] if len(sorted_probs) > 1 else (0, 0)
                
                # Calculate prediction uniqueness (difference between top two)
                uniqueness = top_confidence - second_confidence
                
                # Calculate entropy
                entropy = self.calculate_entropy(probabilities_np)
                
                # Get class name
                if self.label_encoder is not None:
                    original_track_id = self.label_encoder.inverse_transform([top_class_idx])[0]
                    predicted_name = f"Person_{original_track_id}"
                else:
                    predicted_name = self.class_names[top_class_idx]
                
                # Store alternative predictions for conflict resolution
                alternatives = []
                for i, (class_idx, prob) in enumerate(sorted_probs[:3]):
                    if self.label_encoder is not None:
                        alt_track_id = self.label_encoder.inverse_transform([class_idx])[0]
                        alt_name = f"Person_{alt_track_id}"
                    else:
                        alt_name = self.class_names[class_idx]
                    alternatives.append((alt_name, prob))
                
                self.alternative_identities[track_id] = alternatives
                
                # Check for identity conflicts and apply resolution
                final_prediction = self.resolve_identity_conflict(
                    track_id, predicted_name, top_confidence, uniqueness, alternatives
                )
                
                # Store prediction
                self.track_predictions[track_id].append(final_prediction)
                self.track_confidences[track_id].append(top_confidence)
                self.track_all_probabilities[track_id].append(probabilities_np)
                
                # Update stats
                self.debug_stats['total_predictions'] += 1
                self.debug_stats['class_distribution'][final_prediction] += 1
                
                # Update stable identity
                self.update_stable_identity(track_id)
                
                # Debug output for first few predictions
                if self.debug_stats['total_predictions'] <= 5:
                    print(f"🔍 Track {track_id}: {final_prediction} (conf: {top_confidence:.3f}, "
                          f"uniqueness: {uniqueness:.3f}, entropy: {entropy:.3f})")
                    print(f"   Top 3 probs: {alternatives}")
                
        except Exception as e:
            self.debug_stats['prediction_errors'] += 1

    def resolve_identity_conflict(self, track_id, predicted_name, confidence, uniqueness, alternatives):
        """Resolve identity conflicts by assigning alternative identities"""
        
        if not self.conflict_resolution_enabled:
            return predicted_name
        
        # Check if this identity is already assigned to another active track
        if predicted_name in self.identity_assignments:
            existing_track = self.identity_assignments[predicted_name]
            
            # Check if existing track is still active
            if (existing_track in self.stable_identities and 
                existing_track != track_id and
                self.stable_identities[existing_track]['status'] in ['identified', 'likely']):
                
                self.debug_stats['identity_conflicts'] += 1
                
                # Compare confidences
                existing_confidence = self.stable_identities[existing_track]['confidence']
                
                if confidence > existing_confidence + 0.05:  # Clear winner
                    # Reassign to current track
                    print(f"🔄 Reassigning {predicted_name} from Track {existing_track} to Track {track_id}")
                    print(f"   Reason: Higher confidence ({confidence:.3f} vs {existing_confidence:.3f})")
                    
                    # Demote existing track to alternative identity
                    self._demote_track_to_alternative(existing_track)
                    self.identity_assignments[predicted_name] = track_id
                    self.debug_stats['resolved_conflicts'] += 1
                    return predicted_name
                    
                else:
                    # Find alternative identity for current track
                    alternative_name = self._find_alternative_identity(track_id, alternatives)
                    if alternative_name != predicted_name:
                        print(f"🔄 Assigning alternative identity {alternative_name} to Track {track_id}")
                        print(f"   Reason: {predicted_name} already assigned to Track {existing_track}")
                        
                        # Assign alternative if available
                        if alternative_name not in self.identity_assignments:
                            self.identity_assignments[alternative_name] = track_id
                            self.debug_stats['resolved_conflicts'] += 1
                            return alternative_name
                        else:
                            # Create unique identity
                            unique_name = f"Person_Unknown_{track_id}"
                            self.identity_assignments[unique_name] = track_id
                            return unique_name
        else:
            # Identity not assigned, check uniqueness
            if uniqueness < self.uniqueness_threshold:
                # Low uniqueness, consider alternative
                alternative_name = self._find_alternative_identity(track_id, alternatives)
                if alternative_name != predicted_name and alternative_name not in self.identity_assignments:
                    self.identity_assignments[alternative_name] = track_id
                    return alternative_name
            
            # Assign predicted identity
            self.identity_assignments[predicted_name] = track_id
            return predicted_name

    def _demote_track_to_alternative(self, track_id):
        """Demote a track to use alternative identity"""
        if track_id in self.alternative_identities:
            alternatives = self.alternative_identities[track_id]
            # Find first unassigned alternative
            for alt_name, _ in alternatives[1:]:  # Skip first (current) prediction
                if alt_name not in self.identity_assignments:
                    self.identity_assignments[alt_name] = track_id
                    # Update stable identity
                    if track_id in self.stable_identities:
                        self.stable_identities[track_id]['name'] = alt_name
                        self.stable_identities[track_id]['status'] = 'reassigned'
                    return
        
        # No alternative found, create unique identity
        unique_name = f"Person_Unknown_{track_id}"
        self.identity_assignments[unique_name] = track_id
        if track_id in self.stable_identities:
            self.stable_identities[track_id]['name'] = unique_name
            self.stable_identities[track_id]['status'] = 'reassigned'

    def _find_alternative_identity(self, track_id, alternatives):
        """Find the best alternative identity that's not already assigned"""
        for alt_name, confidence in alternatives:
            if alt_name not in self.identity_assignments:
                return alt_name
        
        # No unassigned alternative found, create unique identity
        return f"Person_Unknown_{track_id}"

    def calculate_entropy(self, probabilities):
        """Calculate entropy of probability distribution"""
        eps = 1e-8
        probabilities = np.clip(probabilities, eps, 1.0)
        return -np.sum(probabilities * np.log(probabilities))

    # Replace the update_stable_identity method with this fixed version:

    def update_stable_identity(self, track_id):
        """Update stable identity with enhanced logic"""
        predictions = self.track_predictions[track_id]
        confidences = self.track_confidences[track_id]
        probabilities = self.track_all_probabilities[track_id]
        
        if len(predictions) < self.min_predictions_for_stability:
            return
        
        # Get recent predictions for stability check
        recent_predictions = predictions[-self.min_predictions_for_stability:]
        recent_confidences = confidences[-self.min_predictions_for_stability:]
        recent_probabilities = probabilities[-self.min_predictions_for_stability:]
        
        # Filter out None predictions
        valid_predictions = [p for p in recent_predictions if p is not None]
        if not valid_predictions:
            return
        
        # Check consistency
        prediction_counts = Counter(valid_predictions)
        most_common_prediction, count = prediction_counts.most_common(1)[0]
        
        consistency_ratio = count / len(valid_predictions)
        avg_confidence = np.mean([conf for pred, conf in zip(recent_predictions, recent_confidences) 
                                if pred == most_common_prediction and pred is not None])
        
        # Calculate average entropy
        avg_entropy = np.mean([self.calculate_entropy(prob) for prob in recent_probabilities])
        
        # Determine status with enhanced thresholds
        is_confident = avg_confidence >= self.confidence_threshold
        is_consistent = consistency_ratio >= self.consistency_threshold
        is_certain = avg_entropy <= self.entropy_threshold
        
        if is_confident and is_consistent and is_certain:
            status = 'identified'
        elif is_consistent and is_confident:
            status = 'likely'
        elif is_consistent:
            status = 'maybe'
        else:
            status = 'uncertain'
        
        self.stable_identities[track_id] = {
            'name': most_common_prediction,  # This will now properly show the assigned identity
            'confidence': avg_confidence,
            'consistency': consistency_ratio,
            'entropy': avg_entropy,
            'status': status,
            'prediction_count': len(predictions)
        }

    def get_identification_result(self, track_id):
        """Get current identification result"""
        if track_id in self.stable_identities:
            result = self.stable_identities[track_id]
            return result['name'], result['confidence'], result['status']
        elif track_id in self.track_states:
            state = self.track_states[track_id]
            progress = len(self.track_features[track_id])
            return None, 0.0, f"{state} ({progress}/{self.sequence_length})"
        else:
            return None, 0.0, "new"

    def print_debug_summary(self):
        """Print comprehensive debug summary"""
        print(f"\n=== Enhanced Debug Summary ===")
        print(f"Total predictions: {self.debug_stats['total_predictions']}")
        print(f"Feature extraction errors: {self.debug_stats['feature_extraction_errors']}")
        print(f"Prediction errors: {self.debug_stats['prediction_errors']}")
        print(f"Identity conflicts detected: {self.debug_stats['identity_conflicts']}")
        print(f"Conflicts resolved: {self.debug_stats['resolved_conflicts']}")
        print(f"Class distribution: {dict(self.debug_stats['class_distribution'])}")
        if self.debug_stats['feature_counts']:
            avg_features = np.mean(self.debug_stats['feature_counts'])
            print(f"Average features extracted: {avg_features:.1f}/41")
        
        print(f"\nCurrent identity assignments:")
        for identity, track_id in self.identity_assignments.items():
            if track_id in self.stable_identities:
                status = self.stable_identities[track_id]['status']
                confidence = self.stable_identities[track_id]['confidence']
                print(f"  {identity} -> Track {track_id} ({status}, conf: {confidence:.3f})")

def main():
    """Enhanced inference main function with conflict resolution"""
    parser = argparse.ArgumentParser(description="Enhanced Gait Recognition with Conflict Resolution")
    
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--model", required=True, help="Trained model path (.pth)")
    parser.add_argument("--output", help="Output video file")
    parser.add_argument("--sequence_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--confidence_threshold", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Initialize enhanced inference
    print("🚀 Starting Enhanced Gait Recognition with Conflict Resolution...")
    gait_inference = EnhancedGaitInferenceWithConflictResolution(args.model, args.sequence_length)
    gait_inference.confidence_threshold = args.confidence_threshold
    
    # Initialize detection components
    device = get_best_device()
    
    detector_tracker = DetectionTracker(device=device, pose_device=device)
    pose_analyzer = PoseAnalyzer(
        model_pose=detector_tracker.model_pose,
        pose_device=device,
        history_length=5
    )
    visualizer = Visualizer()
    
    # Process video
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    
    # Setup video writer
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    else:
        out = None
    
    frame_count = 0
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track people
        tracked_detections = detector_tracker.detect_and_track(frame)
        
        # Process each detection
        for detection in tracked_detections:
            if (not hasattr(detection, 'track_id') or 
                detection.track_id is None or 
                detection.track_id <= 0):
                continue
            
            track_id = detection.track_id
            
            # Process pose
            pose_results = pose_analyzer.process_pose(frame, detection, frame_count)
            
            if pose_results:
                # Get keypoints for gait feature extraction
                keypoints = pose_results[0].get('keypoints') if pose_results else None
                
                # Calculate frame features
                features_dict = pose_analyzer.calculate_frame_features(track_id, frame_count)
                
                # Update inference system with all available data
                gait_inference.update_track(
                    track_id, 
                    features_dict, 
                    keypoints, 
                    frame_count, 
                    detection.bbox
                )
            
            # Get identification result
            name, confidence, status = gait_inference.get_identification_result(track_id)
            
            # Enhanced visualization with conflict resolution indicators
            color = visualizer.get_color(track_id)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Status-based text display with conflict resolution indicators
            if status == 'identified':
                text = f"✓ {track_id}: {name} ({confidence:.2f})"
                text_color = (0, 255, 0)  # Green
            elif status == 'likely':
                text = f"⚡ {track_id}: {name} ({confidence:.2f})"
                text_color = (0, 255, 255)  # Yellow
            elif status == 'reassigned':
                text = f"🔄 {track_id}: {name} ({confidence:.2f})"
                text_color = (255, 165, 0)  # Orange
            elif status == 'maybe':
                text = f"? {track_id}: {name} ({confidence:.2f})"
                text_color = (0, 165, 255)  # Light Blue
            elif status == 'uncertain':
                text = f"?? {track_id}: {name} ({confidence:.2f})"
                text_color = (0, 100, 255)  # Red-Orange
            else:
                text = f"○ {track_id}: {status}"
                text_color = (255, 255, 255)  # White
            
            # Draw text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, text_color, thickness)
            
            # Draw pose if available
            if pose_results:
                for pose_result in pose_results:
                    keypoints = pose_result['keypoints']
                    crop_offset = pose_result['crop_offset']
                    visualizer.draw_keypoints(frame, keypoints, crop_offset[0], crop_offset[1], color)
        
        # Save/display frame
        if out is not None:
            out.write(frame)
        
        if args.debug:
            cv2.imshow('Enhanced Gait Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print debug summary
    gait_inference.print_debug_summary()
    
    # Print final results
    print("\n=== Final Enhanced Identification Results ===")
    for track_id in sorted(gait_inference.stable_identities.keys()):
        result = gait_inference.stable_identities[track_id]
        print(f"Track {track_id}: {result['name']} (conf: {result['confidence']:.3f}, "
              f"consistency: {result['consistency']:.3f}, entropy: {result['entropy']:.3f}, "
              f"status: {result['status']}, predictions: {result['prediction_count']})")

if __name__ == "__main__":
    main()