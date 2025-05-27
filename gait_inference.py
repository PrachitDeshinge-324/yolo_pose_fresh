"""
Real-time Gait Recognition Inference

This script uses the same data processing pipeline as main.py to extract gait features
and then applies a trained LSTM model to identify people in real-time.
"""

import argparse
import os
import torch
import numpy as np
import cv2
import pickle
from collections import defaultdict, deque
import json

# Import our modular components (same as main.py)
from detection_tracking import DetectionTracker
from pose_analysis import PoseAnalyzer
from data_processing import DataProcessor

# Import utilities
from utils.visualization import Visualizer

# Import the LSTM model from training script
from train_lstm_gait import BidirectionalLSTM, get_best_device

class GaitInference:
    """Real-time gait recognition inference system"""
    
    def __init__(self, model_path, config_path, sequence_length=20):
        self.device = get_best_device()
        self.sequence_length = sequence_length
        
        # Load trained model and configuration
        self.load_model(model_path, config_path)
        
        # 🚨 CRITICAL: Initialize gait feature extractor for 41-feature extraction
        from utils.skeleton_gait import NormalizedGaitFeatureExtractor
        self.gait_extractor = NormalizedGaitFeatureExtractor()
        
        # Initialize feature buffers for each track
        self.track_features = defaultdict(lambda: deque(maxlen=sequence_length))
        self.track_predictions = defaultdict(list)
        self.track_confidences = defaultdict(list)
        
        # Track state management
        self.track_states = {}  # 'collecting', 'ready', 'identified'
        self.identification_results = {}
        
        print(f"✅ Gait inference system initialized")
        print(f"   Device: {self.device}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Model classes: {self.class_names}")
        print(f"   🔧 Gait feature extractor initialized for 41-feature extraction")
    
    def load_model(self, model_path, config_path):
        """Load the trained model and configuration"""
        print(f"Loading model from {model_path}...")
        
        # Load checkpoint with proper error handling
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
        
        # Get model configuration from checkpoint (not external config)
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            self.sequence_length = model_config['sequence_length']
            hidden_size = model_config['hidden_size']
            num_layers = model_config['num_layers']
            dropout = model_config['dropout']
            
            # Get number of classes from saved model state
            fc3_weight_shape = checkpoint['model_state_dict']['fc3.weight'].shape
            num_classes = fc3_weight_shape[0]  # First dimension is num_classes
            
            # Get input size from model state
            input_norm_weight_shape = checkpoint['model_state_dict']['input_norm.weight'].shape
            input_size = input_norm_weight_shape[0]  # Feature dimension
            self.input_size = input_size  # ADD THIS LINE
            
            print(f"🔍 Detected model dimensions:")
            print(f"   Input size: {input_size}")
            print(f"   Number of classes: {num_classes}")
            print(f"   Hidden size: {hidden_size}")
            print(f"   Sequence length: {self.sequence_length}")
            
        else:
            # Fallback: try to infer from model state dict
            fc3_weight_shape = checkpoint['model_state_dict']['fc3.weight'].shape
            num_classes = fc3_weight_shape[0]
            input_norm_weight_shape = checkpoint['model_state_dict']['input_norm.weight'].shape
            input_size = input_norm_weight_shape[0]
            self.input_size = input_size  # ADD THIS LINE TOO
            
            # Use default values
            hidden_size = 64
            num_layers = 2
            dropout = 0.2
            
            print(f"⚠️  Using inferred dimensions:")
            print(f"   Input size: {input_size}")
            print(f"   Number of classes: {num_classes}")
        
        # Create model with correct dimensions
        from train_lstm_gait import BidirectionalLSTM
        self.model = BidirectionalLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load label encoder
        if 'label_encoder' in checkpoint:
            self.label_encoder = checkpoint['label_encoder']
            # Create class names from label encoder
            self.class_names = [f"Person_{track_id}" for track_id in self.label_encoder.classes_]
            print(f"✓ Loaded label encoder with classes: {self.label_encoder.classes_}")
        else:
            print("⚠️  No label encoder found in checkpoint")
            self.label_encoder = None
            # Create default class names
            self.class_names = [f"Person_{i+1}" for i in range(num_classes)]

        # Load feature scaler
        scaler_path = os.path.join(os.path.dirname(model_path), 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            import joblib
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Loaded feature scaler from {scaler_path}")
        else:
            print(f"⚠️  Feature scaler not found at {scaler_path}")
            self.scaler = None
        
        print(f"✓ Model loaded successfully!")
        return True

    def update_track_features(self, track_id, features_dict, keypoints=None, frame_idx=None):
        """Update feature buffer for a track with comprehensive gait features"""
        if not features_dict:
            return
        
        # 🚨 CRITICAL: If we have keypoints, update the gait extractor first
        if keypoints is not None and frame_idx is not None:
            self.update_track_keypoints(track_id, keypoints, frame_idx)
        
        # Extract comprehensive features (41 features total)
        comprehensive_features = self.extract_comprehensive_features(track_id, features_dict)
        
        # Convert to feature vector (same order as training)
        feature_vector = self.dict_to_vector(comprehensive_features)
        
        if feature_vector is not None:
            # Add to buffer
            self.track_features[track_id].append(feature_vector)
            
            # Update track state
            if len(self.track_features[track_id]) < self.sequence_length:
                self.track_states[track_id] = 'collecting'
            else:
                self.track_states[track_id] = 'ready'
    
    def dict_to_vector(self, features_dict):
        """Convert features dictionary to vector with correct size - FIXED for 41 features"""
        try:
            # 🚨 CRITICAL: This must match the exact feature order used in training!
            # Based on analysis of 1c_frame_features.csv, we need exactly these 41 features in order
            
            # Define the EXACT 41 features used in training (from actual CSV header, excluding metadata columns)
            expected_features = [
                'frame_center_x', 'frame_center_y', 'frame_width', 'frame_height',
                'frame_body_height', 'num_valid_keypoints', 'left_knee_angle', 'right_knee_angle',
                'frame_avg_displacement', 'frame_max_displacement', 'frame_displacement_std',
                # NOTE: 'interpolated' column was excluded during training as metadata
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
            
            # Ensure we have exactly 41 features (matching the trained model)
            assert len(expected_features) == 41, f"Expected 41 features, got {len(expected_features)}"
            
            # Extract features in the exact order
            feature_vector = []
            missing_features = []
            
            for feat_name in expected_features:
                if feat_name in features_dict:
                    value = features_dict[feat_name]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(float(value))
                    else:
                        feature_vector.append(0.0)
                        missing_features.append(feat_name)
                else:
                    feature_vector.append(0.0)
                    missing_features.append(feat_name)
            
            if missing_features and len(missing_features) > 20:  # Only warn if many features missing
                print(f"⚠️  Missing {len(missing_features)}/{len(expected_features)} features: {missing_features[:5]}...")
            
            # Ensure exactly 41 features (matching trained model)
            if len(feature_vector) != 41:
                print(f"🚨 CRITICAL: Feature vector has {len(feature_vector)} features, expected 41!")
                # Pad or truncate to 41
                if len(feature_vector) > 41:
                    feature_vector = feature_vector[:41]
                else:
                    feature_vector.extend([0.0] * (41 - len(feature_vector)))
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error converting features to vector: {e}")
            print(f"Available features: {list(features_dict.keys())}")
            # Return a zero vector of the correct size as fallback
            return np.zeros(self.input_size, dtype=np.float32)
    
    def debug_features(self, features_dict):
        """Debug function to show available features"""
        print(f"\n🔍 Available features ({len(features_dict)} total):")
        for i, (key, value) in enumerate(features_dict.items()):
            if isinstance(value, (int, float)):
                print(f"  {i:2d}. {key}: {value:.4f}")
            elif isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    print(f"  {i:2d}. {key}: [{len(value)} values] mean={np.mean(value):.4f}")
                else:
                    print(f"  {i:2d}. {key}: [empty array]")
            else:
                print(f"  {i:2d}. {key}: {type(value).__name__}")
        print()
    
    def log_available_features(self, features_dict):
        """Log all available feature names for debugging"""
        if features_dict:
            print(f"\n🔍 DEBUG: Generated feature names:")
            feature_names = sorted(features_dict.keys())
            for i, name in enumerate(feature_names):
                print(f"  {i:2d}. '{name}'")
            print(f"Total: {len(feature_names)} features\n")
    
    def predict_identity(self, track_id):
        """Predict identity for a track with sufficient features"""
        if track_id not in self.track_features or len(self.track_features[track_id]) < self.sequence_length:
            return None, 0.0
        
        try:
            # Get feature sequence
            feature_sequence = np.array(list(self.track_features[track_id]))
            
            # Normalize features if scaler is available
            if self.scaler is not None:
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
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_class = predicted_class.cpu().item()
                confidence = confidence.cpu().item()
                
                # Get class name
                if predicted_class < len(self.class_names):
                    predicted_name = self.class_names[predicted_class]
                else:
                    predicted_name = f"Unknown_{predicted_class}"
                
                return predicted_name, confidence
                
        except Exception as e:
            print(f"Error in prediction for track {track_id}: {e}")
            return None, 0.0
    
    def get_track_status(self, track_id):
        """Get current status of a track"""
        if track_id not in self.track_states:
            return "new", 0
        
        state = self.track_states[track_id]
        progress = len(self.track_features[track_id])
        
        return state, progress
    
    def get_identification_result(self, track_id, min_confidence=0.7, min_predictions=3):
        """Get stable identification result for a track"""
        if track_id not in self.track_predictions:
            return None, 0.0, "collecting"
        
        predictions = self.track_predictions[track_id]
        confidences = self.track_confidences[track_id]
        
        if len(predictions) < min_predictions:
            return None, 0.0, "collecting"
        
        # Get most recent predictions
        recent_predictions = predictions[-min_predictions:]
        recent_confidences = confidences[-min_predictions:]
        
        # Check for consistency
        unique_predictions = set(recent_predictions)
        if len(unique_predictions) == 1:
            # Consistent prediction
            predicted_name = recent_predictions[0]
            avg_confidence = np.mean(recent_confidences)
            
            if avg_confidence >= min_confidence:
                return predicted_name, avg_confidence, "identified"
            else:
                return predicted_name, avg_confidence, "low_confidence"
        else:
            # Inconsistent predictions
            return None, 0.0, "inconsistent"
    
    def update_track_keypoints(self, track_id, keypoints, frame_idx):
        """Update keypoints for gait feature extraction"""
        if keypoints is not None and len(keypoints) > 0:
            # Update the gait extractor with raw keypoints
            self.gait_extractor.update_track(track_id, keypoints, frame_idx)
    
    def extract_comprehensive_features(self, track_id, basic_features):
        """Extract comprehensive 42-feature gait features for a track"""
        try:
            # Get normalized gait features from the extractor
            gait_features = self.gait_extractor.get_features(track_id)
            
            # Start with basic frame features (first 12 features)
            comprehensive_features = {}
            
            # Add basic features (frame-level features)
            if basic_features:
                comprehensive_features.update(basic_features)
            
            # 🚨 CRITICAL FIX: Generate exactly the 30 norm_norm_ features needed
            # Map gait extractor features (norm_*) to training data names (norm_norm_*)
            if gait_features:
                # Define the exact mapping from extractor features to training features
                feature_mapping = {
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
                
                # Apply the mapping
                for extractor_name, training_name in feature_mapping.items():
                    if extractor_name in gait_features:
                        comprehensive_features[training_name] = gait_features[extractor_name]
            
            # Debug: Show what features we have
            norm_norm_count = len([k for k in comprehensive_features.keys() if k.startswith('norm_norm_')])
            basic_count = len([k for k in comprehensive_features.keys() if not k.startswith('norm_norm_')])
            
            print(f"📊 Track {track_id} features: {basic_count} basic, {norm_norm_count} norm_norm_ = {len(comprehensive_features)} total")
            
            return comprehensive_features
            
        except Exception as e:
            print(f"Error extracting comprehensive features for track {track_id}: {e}")
            return basic_features if basic_features else {}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Real-time Gait Recognition Inference")
    
    # Input/Output
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", required=True, help="Path to trained LSTM model (.pth)")
    parser.add_argument("--config", help="Path to model config file (auto-detected if not provided)")
    parser.add_argument("--output", help="Path to output video file (optional)")
    parser.add_argument("--results_dir", type=str, default="inference_results", 
                       help="Directory to save inference results")
    
    # Model settings
    parser.add_argument("--use_transreid", action="store_true", default=False,
                       help="Use TransReID for person tracking")
    parser.add_argument("--transreid_model", type=str, default="weights/transreid_vitbase.pth",
                       help="Path to TransReID model weights")
    
    # Inference parameters
    parser.add_argument("--sequence_length", type=int, default=20,
                       help="Sequence length for LSTM input")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                       help="Minimum confidence for stable identification")
    parser.add_argument("--min_predictions", type=int, default=3,
                       help="Minimum consistent predictions for identification")
    
    # Processing options
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Start processing from this frame number")
    parser.add_argument("--end_frame", type=int, default=0,
                       help="End processing at this frame number (0 for full video)")
    
    # Display options
    parser.add_argument("--display", action="store_true", default=True,
                       help="Display video during processing")
    parser.add_argument("--save_video", action="store_true", default=True,
                       help="Save processed video with identifications")
    
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

def draw_identification_bbox(frame, bbox, track_id, color, state, progress, 
                           stable_name, stable_conf, status, max_progress):
    """Draw bounding box with identification information"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text information
    if status == "identified":
        main_text = f"ID {track_id}: {stable_name}"
        conf_text = f"Conf: {stable_conf:.2f}"
        text_color = (0, 255, 0)  # Green for identified
    elif status == "low_confidence":
        main_text = f"ID {track_id}: {stable_name}?"
        conf_text = f"Conf: {stable_conf:.2f} (Low)"
        text_color = (0, 255, 255)  # Yellow for low confidence
    elif status == "inconsistent":
        main_text = f"ID {track_id}: Analyzing..."
        conf_text = "Inconsistent"
        text_color = (0, 165, 255)  # Orange for inconsistent
    else:
        main_text = f"ID {track_id}: Collecting..."
        conf_text = f"Progress: {progress}/{max_progress}"
        text_color = (255, 255, 255)  # White for collecting
    
    # Draw text background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Main text
    (text_width, text_height), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
    cv2.putText(frame, main_text, (x1, y1 - 5), font, font_scale, text_color, thickness)
    
    # Confidence/progress text
    (conf_width, conf_height), _ = cv2.getTextSize(conf_text, font, font_scale - 0.1, thickness - 1)
    cv2.rectangle(frame, (x1, y2), (x1 + conf_width, y2 + conf_height + 5), (0, 0, 0), -1)
    cv2.putText(frame, conf_text, (x1, y2 + conf_height), font, font_scale - 0.1, text_color, thickness - 1)
    
    # Progress bar for collecting state
    if state == "collecting":
        bar_width = 100
        bar_height = 5
        bar_x = x1
        bar_y = y2 + conf_height + 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int((progress / max_progress) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color, -1)

def main():
    """Main inference pipeline with direct video processing"""
    args = parse_args()
    
    # 🔥 Verify video file
    print(f"🎬 Checking video file: {args.video}")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Auto-detect config file if not provided
    if not args.config:
        model_dir = os.path.dirname(args.model)
        args.config = os.path.join(model_dir, 'config.json')
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Setup devices
    device, pose_device = get_device_config()
    
    # Initialize inference system
    print("Initializing inference system...")
    gait_inference = GaitInference(args.model, args.config, args.sequence_length)
    
    # Initialize components (same as main.py)
    print("Initializing detection and pose analysis...")
    
    # 1. Detection and Tracking
    detector_tracker = DetectionTracker(
        device=device,
        pose_device=pose_device,
        use_transreid=args.use_transreid,
        transreid_model=args.transreid_model,
        tracking_iou=0.5,
        tracking_age=30
    )
    
    # 2. Pose Analysis
    pose_analyzer = PoseAnalyzer(
        model_pose=detector_tracker.model_pose,
        pose_device=pose_device,
        history_length=5
    )
    
    # 3. Visualization
    visualizer = Visualizer()
    
    print("Starting real-time inference...")
    
    # Store results
    frame_results = []
    
    # 🔥 Direct video processing (replaced VideoProcessor)
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {args.video}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {width}x{height}")
    
    # Setup video writer if saving
    if args.save_video and args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Writing output to {args.output}")
    else:
        out = None
    
    frame_count = 0
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame > 0 else total_frames
    
    print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame} total)")
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    
    def process_frame(frame, frame_count, fps):
        """Process each frame and perform real-time identification"""
        
        # 🔥 Add debugging every 30 frames
        debug_frame = frame_count % 30 == 0
        
        if debug_frame:
            print(f"\n=== Frame {frame_count} ===")
        
        # 1. Detect and track people
        tracked_detections = detector_tracker.detect_and_track(frame)
        
        if debug_frame:
            print(f"Found {len(tracked_detections)} detections")
        
        # 2. Process each detection
        for detection in tracked_detections:
            if not hasattr(detection, 'track_id') or detection.track_id <= 0:
                continue
            
            track_id = detection.track_id
            
            if debug_frame:
                print(f"  Processing track {track_id}")
            
            try:
                # Process pose and get features
                pose_results = pose_analyzer.process_pose(frame, detection, frame_count)
                
                if pose_results:
                    if debug_frame:
                        print(f"    ✅ Pose: {len(pose_results)} results")
                    
                    # 🚨 CRITICAL: Extract keypoints for gait feature extraction
                    keypoints = None
                    if pose_results and len(pose_results) > 0:
                        keypoints = pose_results[0].get('keypoints')  # Get first pose result's keypoints
                    
                    # Calculate basic frame features
                    features_dict = pose_analyzer.calculate_frame_features(track_id, frame_count)
                    
                    if features_dict:
                        if debug_frame:
                            print(f"    ✅ Basic features: {len(features_dict)} extracted")
                        
                        # 🚨 CRITICAL: Update inference system with keypoints for comprehensive feature extraction
                        gait_inference.update_track_features(track_id, features_dict, keypoints, frame_count)
                        
                        # Check track status
                        state, progress = gait_inference.get_track_status(track_id)
                        
                        if debug_frame:
                            print(f"    📊 Track status: {state}, progress: {progress}/{gait_inference.sequence_length}")
                        
                        # Try prediction if ready
                        if gait_inference.track_states.get(track_id) == 'ready':
                            if debug_frame:
                                print(f"    🎯 Making prediction for track {track_id}")
                            
                            predicted_name, confidence = gait_inference.predict_identity(track_id)
                            
                            if predicted_name:
                                if debug_frame:
                                    print(f"    ✅ Prediction: {predicted_name} (conf: {confidence:.3f})")
                                
                                # Store prediction
                                gait_inference.track_predictions[track_id].append(predicted_name)
                                gait_inference.track_confidences[track_id].append(confidence)
                                
                                # Get stable identification
                                stable_name, stable_conf, status = gait_inference.get_identification_result(
                                    track_id, args.min_confidence, args.min_predictions
                                )
                                
                                # Store results
                                frame_results.append({
                                    'frame': frame_count,
                                    'track_id': track_id,
                                    'predicted_name': predicted_name,
                                    'confidence': confidence,
                                    'stable_name': stable_name,
                                    'stable_confidence': stable_conf,
                                    'status': status
                                })
                    else:
                        if debug_frame:
                            print(f"    ❌ No features extracted")
                else:
                    if debug_frame:
                        print(f"    ❌ No pose results")
            
            except Exception as e:
                if debug_frame:
                    print(f"    ❌ Processing error: {e}")
            
            # Visualization
            color = visualizer.get_color(track_id)
            
            # Get track status and identification
            state, progress = gait_inference.get_track_status(track_id)
            stable_name, stable_conf, status = gait_inference.get_identification_result(track_id)
            
            # Draw visualization
            if 'pose_results' in locals() and pose_results:
                for pose_result in pose_results:
                    keypoints = pose_result['keypoints']
                    crop_offset = pose_result['crop_offset']
                    visualizer.draw_keypoints(frame, keypoints, crop_offset[0], crop_offset[1], color)
            
            # Draw enhanced bounding box with identification info
            draw_identification_bbox(
                frame, detection.bbox, track_id, color, 
                state, progress, stable_name, stable_conf, status,
                args.sequence_length
            )
        
        return frame
    
    # 🔥 Main processing loop
    try:
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = process_frame(frame, frame_count, fps)
            
            # Save frame if recording
            if out is not None:
                out.write(processed_frame)
            
            # Display frame
            if args.display:
                cv2.imshow('Gait Recognition Inference', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count - start_frame) / (end_frame - start_frame) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{end_frame})")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    
    print("Inference complete!")
    
    # Save results
    results_file = os.path.join(args.results_dir, "inference_results.json")
    with open(results_file, 'w') as f:
        json.dump(frame_results, f, indent=2)
    
    # Generate summary
    print("\n=== Inference Summary ===")
    unique_tracks = set([r['track_id'] for r in frame_results])
    print(f"Processed {len(unique_tracks)} unique tracks")
    
    for track_id in sorted(unique_tracks):
        track_results = [r for r in frame_results if r['track_id'] == track_id]
        
        # Get final identification
        identified_results = [r for r in track_results if r['status'] == 'identified']
        
        if identified_results:
            final_result = identified_results[-1]
            print(f"Track {track_id}: {final_result['stable_name']} (confidence: {final_result['stable_confidence']:.2f})")
        else:
            print(f"Track {track_id}: Not identified")
    
    print(f"\nResults saved to: {results_file}")
    if args.save_video and args.output:
        print(f"Video saved to: {args.output}")

if __name__ == "__main__":
    main()