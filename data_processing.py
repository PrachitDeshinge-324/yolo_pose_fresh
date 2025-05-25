"""
Data Processing Module

Handles frame-by-frame feature storage with ID merging capability.
Stores features for every frame of every track ID including normalized gait features.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.id_merger import IDMerger

class DataProcessor:
    """Data processing with frame-by-frame feature storage"""
    
    def __init__(self):
        self.bbox_info = defaultdict(list)
        self.frame_features = defaultdict(list)  # track_id -> [(frame_idx, features_dict), ...]
        self.id_to_person = {}   # Track ID to person name mapping
        self.merged_ids = {}     # Merged ID mapping
    
    def collect_bbox_info(self, track_id, bbox, frame_idx):
        """Collect bounding box information for each track"""
        if track_id > 0:
            x1, y1, x2, y2 = (bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            self.bbox_info[track_id].append({
                'frame_idx': frame_idx,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'track_id': int(track_id)
            })
    
    def store_frame_features(self, track_id, frame_idx, features_dict):
        """Store calculated features for a specific frame and track ID"""
        if track_id > 0 and features_dict:
            self.frame_features[track_id].append((frame_idx, features_dict))
    
    def apply_id_merging_to_frame_features(self, merged_ids):
        """Apply ID merging to frame-by-frame features"""
        print(f"Applying ID merging to frame features...")
        print(f"Merge operations: {merged_ids}")
        
        merged_features = defaultdict(list)
        
        # Apply merging: if ID 2 should become ID 1, all ID 2 features go to ID 1
        for track_id, features_list in self.frame_features.items():
            # Check if this track_id should be merged into another
            final_id = merged_ids.get(track_id, track_id)  # Use merged ID if exists, otherwise original
            
            # Add all frame features to the final ID
            merged_features[final_id].extend(features_list)
            
            if track_id != final_id:
                print(f"  Merged ID {track_id} → ID {final_id} ({len(features_list)} feature frames)")
        
        # Sort by frame index for each track
        for track_id in merged_features:
            merged_features[track_id].sort(key=lambda x: x[0])  # Sort by frame_idx
        
        final_merged = dict(merged_features)
        print(f"Frame features merging complete:")
        print(f"  Original tracks: {len(self.frame_features)}")
        print(f"  Final tracks: {len(final_merged)}")
        
        return final_merged
    
    def export_frame_by_frame_features(self, merged_frame_features, args, id_to_name=None, pose_analyzer=None):
        """Export frame-by-frame features to CSV with normalized gait features"""
        print("=== Exporting Frame-by-Frame Features with Normalized Features ===")
        
        # Create output paths
        paths = self._create_output_paths(args)
        
        # Check if we have pose_analyzer for normalized features
        if pose_analyzer and hasattr(pose_analyzer, 'gait_analyzer') and pose_analyzer.gait_analyzer:
            print("Including normalized gait features from pose analyzer...")
            self._save_frame_features_with_normalized(merged_frame_features, paths, id_to_name, pose_analyzer.gait_analyzer)
        else:
            print("No gait analyzer found, saving basic frame features only...")
            self._save_basic_frame_features(merged_frame_features, paths, id_to_name)
        
        # Save metadata files
        self._save_metadata_files(merged_frame_features, args, id_to_name, paths)
        
        # Save track summary
        self._save_track_summary(merged_frame_features, paths, id_to_name)
        
        return paths
    
    def _save_basic_frame_features(self, merged_frame_features, paths, id_to_name):
        """Save basic frame features without normalized features"""
        all_rows = []
        
        for track_id, features_list in merged_frame_features.items():
            person_name = id_to_name.get(str(track_id), f"Person_{track_id}") if id_to_name else f"Person_{track_id}"
            
            for frame_idx, features_dict in features_list:
                # Create row with basic info + all features
                row = {
                    'track_id': track_id,
                    'frame_idx': frame_idx,
                    'person_name': person_name,
                    **features_dict  # Add all calculated features
                }
                all_rows.append(row)
        
        if not all_rows:
            print("No feature data to export!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        df = df.fillna(0)
        
        # Sort by track_id and frame_idx
        df = df.sort_values(['track_id', 'frame_idx'])
        
        # Save complete CSV with all data
        df.to_csv(paths['features_csv'], index=False)
        print(f"✓ Saved basic frame-by-frame CSV: {df.shape}")
        
        # Save NumPy array (only numeric feature columns)
        numeric_columns = [col for col in df.columns if col not in ['track_id', 'frame_idx', 'person_name']]
        if numeric_columns:
            numeric_df = df[numeric_columns]
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0)
            
            numpy_array = numeric_df.values.astype(np.float32)
            np.save(paths['features_npy'], numpy_array)
            print(f"✓ Saved NumPy features: {numpy_array.shape}")
        
        print(f"✓ Exported {len(all_rows)} total feature records for {len(merged_frame_features)} tracks")
    
    def _save_frame_features_with_normalized(self, merged_frame_features, paths, id_to_name, gait_analyzer):
        """Save frame features along with normalized gait features with interpolation for missing frames"""
        all_data = []
        
        for track_id, features_list in merged_frame_features.items():
            person_name = id_to_name.get(str(track_id), f"Person_{track_id}") if id_to_name else f"Person_{track_id}"
            
            # Get normalized features for this track (computed once per track)
            normalized_features = gait_analyzer.get_features(track_id) if gait_analyzer else {}
            invariant_features = gait_analyzer.calculate_view_invariant_features(track_id) if gait_analyzer else {}
            
            # 🔥 Handle None values safely
            if normalized_features is None:
                normalized_features = {}
            if invariant_features is None:
                invariant_features = {}
            
            print(f"Track {track_id}: {len(normalized_features)} normalized features, {len(invariant_features)} invariant features")
            
            # 🔥 NEW: Interpolate missing frames for this track
            interpolated_features = self._interpolate_missing_frames(features_list)
            
            for frame_idx, frame_features in interpolated_features:
                # Start with basic frame data
                row_data = {
                    'track_id': track_id,
                    'frame_idx': frame_idx,
                    'person_name': person_name
                }
                
                # Add frame-level features
                row_data.update({
                    'frame_center_x': frame_features.get('frame_center_x', 0),
                    'frame_center_y': frame_features.get('frame_center_y', 0),
                    'frame_width': frame_features.get('frame_width', 0),
                    'frame_height': frame_features.get('frame_height', 0),
                    'frame_body_height': frame_features.get('frame_body_height', 0),
                    'num_valid_keypoints': frame_features.get('num_valid_keypoints', 0),
                    'left_knee_angle': frame_features.get('left_knee_angle', 0),
                    'right_knee_angle': frame_features.get('right_knee_angle', 0),
                    'frame_avg_displacement': frame_features.get('frame_avg_displacement', 0),
                    'frame_max_displacement': frame_features.get('frame_max_displacement', 0),
                    'frame_displacement_std': frame_features.get('frame_displacement_std', 0),
                    'interpolated': frame_features.get('interpolated', False)  # Mark interpolated frames
                })
                
                # Add normalized gait features (same for all frames of this track)
                if normalized_features:
                    for feature_name, feature_value in normalized_features.items():
                        if isinstance(feature_value, (int, float, np.integer, np.floating)):
                            row_data[f'norm_{feature_name}'] = feature_value
                        elif isinstance(feature_value, (list, np.ndarray)):
                            # Handle array features by taking mean or first value
                            if len(feature_value) > 0:
                                row_data[f'norm_{feature_name}'] = np.mean(feature_value) if len(feature_value) > 1 else feature_value[0]
                            else:
                                row_data[f'norm_{feature_name}'] = 0
                        else:
                            # Handle other types (convert to string or skip)
                            try:
                                row_data[f'norm_{feature_name}'] = float(feature_value)
                            except (ValueError, TypeError):
                                row_data[f'norm_{feature_name}'] = 0
                
                # Add view-invariant features
                if invariant_features:
                    for feature_name, feature_value in invariant_features.items():
                        if isinstance(feature_value, (int, float, np.integer, np.floating)):
                            row_data[f'inv_{feature_name}'] = feature_value
                        elif isinstance(feature_value, (list, np.ndarray)):
                            # Handle array features
                            if len(feature_value) > 0:
                                row_data[f'inv_{feature_name}'] = np.mean(feature_value) if len(feature_value) > 1 else feature_value[0]
                            else:
                                row_data[f'inv_{feature_name}'] = 0
                        else:
                            # Handle other types
                            try:
                                row_data[f'inv_{feature_name}'] = float(feature_value)
                            except (ValueError, TypeError):
                                row_data[f'inv_{feature_name}'] = 0
                
                all_data.append(row_data)
        
        if not all_data:
            print("No feature data to export!")
            return
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        df = df.fillna(0)
        
        # Sort by track_id and frame_idx
        df = df.sort_values(['track_id', 'frame_idx'])
        
        # Save complete CSV with all features
        df.to_csv(paths['features_csv'], index=False)
        print(f"✓ Saved frame-by-frame CSV with normalized features: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Count interpolated frames
        interpolated_count = df['interpolated'].sum() if 'interpolated' in df.columns else 0
        print(f"  Interpolated frames: {interpolated_count}")
        
        # Save NumPy array (only numeric feature columns)
        numeric_columns = [col for col in df.columns if col not in ['track_id', 'frame_idx', 'person_name', 'interpolated']]
        if numeric_columns:
            numeric_df = df[numeric_columns]
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0)
            
            numpy_array = numeric_df.values.astype(np.float32)
            np.save(paths['features_npy'], numpy_array)
            print(f"✓ Saved NumPy features: {numpy_array.shape}")
        
        print(f"✓ Exported {len(all_data)} total feature records with normalized features")

    def _interpolate_missing_frames(self, features_list):
        """Interpolate missing frames using mean of surrounding frames"""
        if len(features_list) <= 1:
            return features_list
        
        # Sort by frame index
        sorted_features = sorted(features_list, key=lambda x: x[0])
        
        # Get frame indices and check for gaps
        frame_indices = [f[0] for f in sorted_features]
        min_frame = min(frame_indices)
        max_frame = max(frame_indices)
        
        # Create a complete sequence
        complete_sequence = []
        feature_dict = {frame_idx: features for frame_idx, features in sorted_features}
        
        print(f"    Interpolating frames {min_frame} to {max_frame} (original: {len(sorted_features)} frames)")
        
        interpolated_count = 0
        
        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx in feature_dict:
                # Frame exists, use original data
                complete_sequence.append((frame_idx, feature_dict[frame_idx]))
            else:
                # Frame missing, interpolate
                interpolated_features = self._interpolate_frame_features(frame_idx, feature_dict, min_frame, max_frame)
                interpolated_features['interpolated'] = True  # Mark as interpolated
                complete_sequence.append((frame_idx, interpolated_features))
                interpolated_count += 1
        
        if interpolated_count > 0:
            print(f"    Interpolated {interpolated_count} missing frames")
        
        return complete_sequence

    def _interpolate_frame_features(self, target_frame, feature_dict, min_frame, max_frame):
        """Interpolate features for a missing frame using surrounding frames"""
        
        # Find closest frames before and after target frame
        before_frame = None
        after_frame = None
        
        # Find closest frame before target
        for frame_idx in range(target_frame - 1, min_frame - 1, -1):
            if frame_idx in feature_dict:
                before_frame = frame_idx
                break
        
        # Find closest frame after target
        for frame_idx in range(target_frame + 1, max_frame + 1):
            if frame_idx in feature_dict:
                after_frame = frame_idx
                break
        
        # Get feature template from any existing frame
        template_features = next(iter(feature_dict.values()))
        interpolated = {}
        
        # Interpolate each feature
        for feature_name in template_features.keys():
            if feature_name == 'interpolated':
                continue
                
            before_value = None
            after_value = None
            
            # Get values from surrounding frames
            if before_frame is not None:
                before_value = feature_dict[before_frame].get(feature_name, 0)
            if after_frame is not None:
                after_value = feature_dict[after_frame].get(feature_name, 0)
            
            # Interpolate value
            if before_value is not None and after_value is not None:
                # Linear interpolation
                frame_distance = after_frame - before_frame
                if frame_distance > 0:
                    weight = (target_frame - before_frame) / frame_distance
                    interpolated_value = before_value + weight * (after_value - before_value)
                else:
                    interpolated_value = before_value
            elif before_value is not None:
                # Use before value
                interpolated_value = before_value
            elif after_value is not None:
                # Use after value
                interpolated_value = after_value
            else:
                # Default value
                interpolated_value = 0
            
            # Ensure numeric type
            try:
                interpolated[feature_name] = float(interpolated_value)
            except (ValueError, TypeError):
                interpolated[feature_name] = 0
        
        return interpolated

    def _save_basic_frame_features(self, merged_frame_features, paths, id_to_name):
        """Save basic frame features without normalized features (with interpolation)"""
        all_rows = []
        
        for track_id, features_list in merged_frame_features.items():
            person_name = id_to_name.get(str(track_id), f"Person_{track_id}") if id_to_name else f"Person_{track_id}"
            
            # 🔥 NEW: Interpolate missing frames for this track
            interpolated_features = self._interpolate_missing_frames(features_list)
            
            for frame_idx, features_dict in interpolated_features:
                # Create row with basic info + all features
                row = {
                    'track_id': track_id,
                    'frame_idx': frame_idx,
                    'person_name': person_name,
                    **features_dict  # Add all calculated features (including 'interpolated' flag)
                }
                all_rows.append(row)
        
        if not all_rows:
            print("No feature data to export!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        df = df.fillna(0)
        
        # Sort by track_id and frame_idx
        df = df.sort_values(['track_id', 'frame_idx'])
        
        # Save complete CSV with all data
        df.to_csv(paths['features_csv'], index=False)
        print(f"✓ Saved basic frame-by-frame CSV: {df.shape}")
        
        # Count interpolated frames
        interpolated_count = df['interpolated'].sum() if 'interpolated' in df.columns else 0
        print(f"  Interpolated frames: {interpolated_count}")
        
        # Save NumPy array (only numeric feature columns)
        numeric_columns = [col for col in df.columns if col not in ['track_id', 'frame_idx', 'person_name', 'interpolated']]
        if numeric_columns:
            numeric_df = df[numeric_columns]
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0)
            
            numpy_array = numeric_df.values.astype(np.float32)
            np.save(paths['features_npy'], numpy_array)
            print(f"✓ Saved NumPy features: {numpy_array.shape}")
        
        print(f"✓ Exported {len(all_rows)} total feature records for {len(merged_frame_features)} tracks")
    def _save_metadata_files(self, merged_frame_features, args, id_to_name, paths):
        """Save metadata and mapping files"""
        
        # 1. Save ID to person mapping
        id_to_person_final = {}
        for track_id in merged_frame_features.keys():
            person_name = id_to_name.get(str(track_id), f"Person_{track_id}") if id_to_name else f"Person_{track_id}"
            id_to_person_final[str(track_id)] = person_name
        
        with open(paths['id_to_person'], 'w') as f:
            json.dump(id_to_person_final, f, indent=2)
        print(f"✓ ID to person mapping: {len(id_to_person_final)} entries")
        
        # 2. Save merged IDs
        with open(paths['merged_ids'], 'w') as f:
            json.dump(self.merged_ids, f, indent=2)
        print(f"✓ Merged IDs: {len(self.merged_ids)} entries")
        
        # 3. Save frame index mapping
        frame_mapping = {}
        for track_id, features_list in merged_frame_features.items():
            frame_mapping[str(track_id)] = [frame_idx for frame_idx, _ in features_list]
        
        frame_mapping_path = os.path.join(args.results_dir, "frame_mapping.json")
        with open(frame_mapping_path, 'w') as f:
            json.dump(frame_mapping, f, indent=2)
        print(f"✓ Frame mapping: {len(frame_mapping)} tracks")
    
    def _save_track_summary(self, merged_frame_features, paths, id_to_name):
        """Save track summary statistics"""
        summary_data = []
        
        for track_id, features_list in merged_frame_features.items():
            person_name = id_to_name.get(str(track_id), f"Person_{track_id}") if id_to_name else f"Person_{track_id}"
            
            frame_indices = [frame_idx for frame_idx, _ in features_list]
            
            summary_data.append({
                'track_id': track_id,
                'person_name': person_name,
                'total_frames': len(features_list),
                'first_frame': min(frame_indices) if frame_indices else 0,
                'last_frame': max(frame_indices) if frame_indices else 0,
                'frame_span': max(frame_indices) - min(frame_indices) + 1 if frame_indices else 0
            })
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(os.path.dirname(paths['features_csv']), "track_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Track summary: {len(summary_data)} tracks")
    
    def export_temp_data_for_merger(self, args):
        """Export temporary data for ID merger (using existing frame features)"""
        print("=== Exporting Temporary Data for ID Merger ===")
        os.makedirs(args.results_dir, exist_ok=True)
        
        # 1. Export bounding boxes (needed for ID merger visualization)
        bbox_path = os.path.join(args.results_dir, "temp_bbox_info.json")
        self._save_bbox_info(bbox_path)
        
        # 2. Create aggregated features for similarity comparison
        similarity_features_path = self._create_aggregated_features_for_similarity(args)
        
        print(f"Temporary data exported to {args.results_dir}")
        return {
            'bbox_json': bbox_path,
            'similarity_features_npy': similarity_features_path
        }
    
    def _create_aggregated_features_for_similarity(self, args):
        """Create aggregated features from frame features for similarity comparison"""
        print("Creating aggregated features for similarity comparison...")
        
        aggregated_data = []
        track_ids = []
        
        for track_id in sorted(self.frame_features.keys()):
            if track_id > 0:
                features_list = self.frame_features[track_id]
                
                if not features_list:
                    continue
                
                # Aggregate features across all frames for this track
                all_feature_values = defaultdict(list)
                
                for frame_idx, features_dict in features_list:
                    for feature_name, feature_value in features_dict.items():
                        if isinstance(feature_value, (int, float)):
                            all_feature_values[feature_name].append(feature_value)
                
                # Calculate mean values for similarity
                aggregated_vector = []
                for feature_name in sorted(all_feature_values.keys()):
                    values = all_feature_values[feature_name]
                    if values:
                        mean_value = np.mean(values)
                        aggregated_vector.append(mean_value)
                
                if aggregated_vector:
                    aggregated_data.append(aggregated_vector)
                    track_ids.append(track_id)
        
        # Save aggregated features for similarity
        if aggregated_data:
            similarity_path = os.path.join(args.results_dir, "temp_similarity_features.npy")
            similarity_array = np.array(aggregated_data, dtype=np.float32)
            np.save(similarity_path, similarity_array)
            
            # Save track ID mapping
            ids_path = os.path.join(args.results_dir, "temp_track_ids.json")
            with open(ids_path, 'w') as f:
                json.dump(track_ids, f)
            
            print(f"Created similarity features: {similarity_array.shape}")
            return similarity_path
        
        return None
    
    def merge_ids(self, args, temp_paths):
        """Interactive session to merge incorrectly split tracking IDs"""
        print("\n=== Starting ID Merger ===")
        
        # Extract paths from temp_paths dictionary
        if isinstance(temp_paths, dict):
            bbox_json_path = temp_paths.get('bbox_json')
            features_path = temp_paths.get('similarity_features_npy')
        else:
            # Fallback
            bbox_json_path = os.path.join(args.results_dir, "temp_bbox_info.json")
            features_path = temp_paths
        
        if not bbox_json_path or not os.path.exists(bbox_json_path):
            print(f"Error: Bounding box info file not found")
            return None, None
        
        # Use features file for merger similarity calculation
        merger = IDMerger(args.video, bbox_json_path, features_path or bbox_json_path)
        
        try:
            # Run interactive merging session
            merged_ids, id_to_name = merger.merge_ids_interactive()
            
            # Store the mappings
            self.merged_ids = merged_ids or {}
            self.id_to_person = id_to_name or {}
            
            print(f"ID merging complete: {len(self.merged_ids)} merges, {len(self.id_to_person)} named IDs")
            return merged_ids, id_to_name
        
        except Exception as e:
            print(f"Error during ID merging: {e}")
            return None, None
        
        finally:
            merger.close()
            # Clean up temporary files
            if features_path and os.path.exists(features_path):
                try:
                    os.remove(features_path)
                except:
                    pass
    
    def _save_bbox_info(self, bbox_path):
        """Save bounding box information"""
        filtered_bbox_info = {
            str(track_id): self.bbox_info[track_id]
            for track_id in self.bbox_info
            if track_id > 0
        }
        with open(bbox_path, 'w') as f:
            json.dump(filtered_bbox_info, f, indent=2)
        print(f"Saved bounding box info: {len(filtered_bbox_info)} tracks")
    
    def _create_output_paths(self, args):
        """Create standardized output file paths"""
        os.makedirs(args.results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        
        return {
            'id_to_person': os.path.join(args.results_dir, "id_to_person.json"),
            'merged_ids': os.path.join(args.results_dir, "merged_ids.json"),
            'features_npy': os.path.join(args.results_dir, f"{base_name}_frame_features.npy"),
            'features_csv': os.path.join(args.results_dir, f"{base_name}_frame_features.csv"),
            'bbox_json': os.path.join(args.results_dir, "bbox_info.json")
        }
    
    def get_summary_stats(self):
        """Get summary statistics of collected data"""
        stats = {
            'total_tracks': len(self.frame_features),
            'total_feature_records': sum(len(features_list) for features_list in self.frame_features.values()),
            'tracks_detail': {}
        }
        
        for track_id, features_list in self.frame_features.items():
            if features_list:
                frame_indices = [frame_idx for frame_idx, _ in features_list]
                stats['tracks_detail'][track_id] = {
                    'frames': len(features_list),
                    'first_frame': min(frame_indices),
                    'last_frame': max(frame_indices)
                }
        
        return stats