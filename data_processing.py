"""
Data Processing Module

Simplified data export functionality for 4 essential files.
Maintains compatibility with existing merge_ids workflow.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.id_merger import IDMerger

class DataProcessor:
    """Simplified data processing and export handler"""
    
    def __init__(self):
        self.bbox_info = defaultdict(list)
        self.id_to_person = {}  # Track ID to person name mapping
        self.merged_ids = {}    # Merged ID mapping
    
    def collect_bbox_info(self, track_id, bbox, frame_idx):
        """Collect bounding box information for each track (ID merger compatible, int coordinates)"""
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
    
    def merge_ids(self, args, features_npy_path):
        """Interactive session to merge incorrectly split tracking IDs"""
        print("\n=== Starting ID Merger ===")
        print("This utility will help you merge tracking IDs that belong to the same person.")
        
        bbox_json_path = os.path.join(args.results_dir, "bbox_info.json")
        
        if not os.path.exists(bbox_json_path):
            print(f"Error: Bounding box info file not found at {bbox_json_path}")
            return None, None
        
        merger = IDMerger(args.video, bbox_json_path, features_npy_path)
        
        try:
            # Run interactive merging session
            merged_ids, id_to_name = merger.merge_ids_interactive()
            
            # Store the mappings
            self.merged_ids = merged_ids or {}
            self.id_to_person = id_to_name or {}
            
            # Save results
            merger.save_updated_data(args.results_dir)
            
            return merged_ids, id_to_name
        
        finally:
            merger.close()
    
    def create_output_paths(self, args):
        """Create standardized output file paths"""
        os.makedirs(args.results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        
        paths = {
            'id_to_person': os.path.join(args.results_dir, "id_to_person.json"),
            'merged_ids': os.path.join(args.results_dir, "merged_ids.json"),
            'features_npy': os.path.join(args.results_dir, f"{base_name}_features.npy"),
            'features_csv': os.path.join(args.results_dir, f"{base_name}_features.csv"),
            'bbox_json': os.path.join(args.results_dir, "bbox_info.json"),  # For merge_ids compatibility
            'flat_npy': os.path.join(args.results_dir, f"{base_name}_features.npy")  # Legacy compatibility
        }
        
        return paths
    
    def export_complete_dataset(self, gait_analyzer, args):
        """Export the 4 essential files with merge_ids compatibility"""
        print("=== Starting Simple Data Export ===")
        
        # Create output paths
        paths = self.create_output_paths(args)
        
        # Save bounding box info first (needed for merge_ids)
        if args.save_bbox_info:
            self._save_bbox_info(paths['bbox_json'])
        
        # Extract and prepare features data
        features_data, feature_columns = self._extract_features(gait_analyzer)
        
        if not features_data:
            print("No valid feature data found!")
            return paths
        
        # Handle merge_ids workflow
        if args.merge_ids:
            # Create DataFrame first to handle mixed data types properly
            df_temp = pd.DataFrame(features_data, columns=feature_columns)
            df_temp = df_temp.fillna(0)
            
            # Convert all values to float and flatten any nested structures
            for col in df_temp.columns:
                if col not in ['track_id', 'person_name']:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
            
            # Create temporary numpy file for merge_ids (only numeric columns)
            temp_npy_path = os.path.join(args.results_dir, "temp_features.npy")
            numeric_columns = [col for col in df_temp.columns if col not in ['track_id', 'person_name']]
            temp_array = df_temp[numeric_columns].values.astype(np.float32)
            np.save(temp_npy_path, temp_array)
            
            # Run merge_ids interactive session
            merged_ids, id_to_name = self.merge_ids(args, temp_npy_path)
            
            # Clean up temp file
            if os.path.exists(temp_npy_path):
                os.remove(temp_npy_path)
            
            # Update features with merged information
            features_data = self._apply_merging(features_data, merged_ids, id_to_name)
        
        # Save the 4 essential files
        self._save_final_files(features_data, feature_columns, paths)
        
        print("=== Export Complete ===")
        print(f"ID to Person: {paths['id_to_person']}")
        print(f"Merged IDs: {paths['merged_ids']}")
        print(f"Features NPY: {paths['features_npy']}")
        print(f"Features CSV: {paths['features_csv']}")
        
        return paths
        
    def _save_bbox_info(self, bbox_json_path):
        """Save bounding box information for merge_ids compatibility"""
        filtered_bbox_info = {
            track_id: self.bbox_info[track_id]
            for track_id in self.bbox_info
            if track_id > 0
        }
        with open(bbox_json_path, 'w') as f:
            json.dump(filtered_bbox_info, f)
        print(f"Saved bounding box information: {len(filtered_bbox_info)} tracks")
    
    def _extract_features(self, gait_analyzer):
        """Extract features from gait analyzer"""
        print("Extracting features...")
        
        # Filter valid tracks
        valid_tracks = {
            track_id: history 
            for track_id, history in gait_analyzer.track_history.items() 
            if track_id > 0
        }
        
        if not valid_tracks:
            print("No valid tracks found!")
            return [], []
        
        # Collect all features
        all_data = []
        feature_columns = None
        
        for track_id in valid_tracks.keys():
            # Get features for this track
            features = gait_analyzer.get_features(track_id) or {}
            inv_features = gait_analyzer.calculate_view_invariant_features(track_id) or {}
            
            # Combine all features
            combined_features = {}
            combined_features.update(features)
            for key, value in inv_features.items():
                combined_features[f"inv_{key}"] = value
            
            # Create row data
            row_data = {
                'track_id': track_id,
                'person_name': f"Person_{track_id}",  # Default name, will be updated by merge_ids
                **combined_features
            }
            
            all_data.append(row_data)
            
            # Set feature columns from first valid entry
            if feature_columns is None:
                feature_columns = list(row_data.keys())
        
        print(f"Extracted features for {len(all_data)} tracks")
        return all_data, feature_columns
    
    def _apply_merging(self, features_data, merged_ids, id_to_name):
        """Apply ID merging and naming to features data"""
        if not merged_ids and not id_to_name:
            return features_data
        
        print("Applying ID merging and naming...")
        
        # Update person names and handle merged IDs
        for row in features_data:
            track_id = str(row['track_id'])
            
            # Apply person naming
            if id_to_name and track_id in id_to_name:
                row['person_name'] = id_to_name[track_id]
            
            # Handle merged IDs (if a track was merged into another)
            if merged_ids:
                for merged_from, merged_to in merged_ids.items():
                    if track_id == str(merged_from):
                        # This track was merged into another
                        row['merged_into'] = merged_to
                        if id_to_name and str(merged_to) in id_to_name:
                            row['person_name'] = id_to_name[str(merged_to)]
        
        return features_data
    
    def _save_final_files(self, features_data, feature_columns, paths):
        """Save the 4 essential files ONLY"""
        
        # 1. Save ID to person mapping
        id_to_person_final = {}
        for row in features_data:
            id_to_person_final[str(row['track_id'])] = row['person_name']
        
        with open(paths['id_to_person'], 'w') as f:
            json.dump(id_to_person_final, f, indent=2)
        print(f"Saved ID to person mapping: {len(id_to_person_final)} entries")
        
        # 2. Save merged IDs
        with open(paths['merged_ids'], 'w') as f:
            json.dump(self.merged_ids, f, indent=2)
        print(f"Saved merged IDs: {len(self.merged_ids)} entries")
        
        # 3 & 4. Save features (CSV and NumPy)
        # Create DataFrame
        df = pd.DataFrame(features_data, columns=feature_columns)
        
        # Fill NaN values with 0 for consistency
        df = df.fillna(0)
        
        # Save CSV (with all columns including strings)
        df.to_csv(paths['features_csv'], index=False)
        print(f"Saved CSV features: {df.shape}")
        
        # Save NumPy array (ONLY numeric columns, no strings)
        numeric_columns = [col for col in df.columns if col not in ['track_id', 'person_name']]
        numeric_df = df[numeric_columns]
        
        # Convert all numeric values to float32 for consistency
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0)
        
        numpy_array = numeric_df.values.astype(np.float32)
        np.save(paths['features_npy'], numpy_array)
        print(f"Saved NumPy features: {numpy_array.shape}")
        
        # REMOVED: No more column files or extra files
        print("Saved 4 essential files only")
    # Legacy method names for compatibility
    def export_all_data(self, *args, **kwargs):
        """Legacy compatibility method"""
        print("Note: export_all_data is deprecated, use export_complete_dataset instead")
        return []
    
    def generate_processed_csv(self, *args, **kwargs):
        """Legacy compatibility method"""
        print("Note: generate_processed_csv is deprecated, use export_complete_dataset instead")
        return True
    
    def fix_invariant_features(self, *args, **kwargs):
        """Legacy compatibility method"""
        print("Note: fix_invariant_features is deprecated, use export_complete_dataset instead")
        return True