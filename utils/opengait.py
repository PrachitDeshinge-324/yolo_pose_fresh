import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Example: update these paths as needed
DEFAULT_CFG = os.path.join(os.path.dirname(__file__), '../../gaitbase_da_casiab.yaml')
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), '../weights/GaitBase_DA-60000.pt')

class TemporalAttentionModule(nn.Module):
    """Temporal attention to focus on discriminative gait cycles"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv2 = nn.Conv2d(channels // 4, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (N, C, H, W)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class SpatialAttentionModule(nn.Module):
    """Spatial attention to handle occlusions"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction for robust representation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 3, padding=1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 5, padding=2),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 7, padding=3),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            )
        ])
        
    def forward(self, x):
        features = [scale(x) for scale in self.scales]
        return torch.cat(features, dim=1)

class GaitPartitionModule(nn.Module):
    """Partition-based feature extraction to handle body part occlusion"""
    def __init__(self, channels, num_parts=4):
        super().__init__()
        self.num_parts = num_parts
        self.part_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            ) for _ in range(num_parts)
        ])
        
    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        part_height = H // self.num_parts
        
        part_features = []
        for i, part_conv in enumerate(self.part_convs):
            start_h = i * part_height
            end_h = start_h + part_height if i < self.num_parts - 1 else H
            part_x = x[:, :, start_h:end_h, :]
            part_feat = part_conv(part_x).view(N, C)
            part_features.append(part_feat)
            
        return torch.stack(part_features, dim=1)  # (N, num_parts, C)

class AdvancedGaitNet(nn.Module):
    """Advanced gait recognition network for industrial scenarios"""
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        
        # Initial feature extraction
        self.backbone = nn.Sequential(
            MultiScaleFeatureExtractor(in_channels, 64),
            SpatialAttentionModule(),
            nn.MaxPool2d(2),
            
            MultiScaleFeatureExtractor(64, 128),
            SpatialAttentionModule(),
            nn.MaxPool2d(2),
            
            MultiScaleFeatureExtractor(128, 256),
            TemporalAttentionModule(256),
            nn.MaxPool2d(2),
            
            MultiScaleFeatureExtractor(256, feature_dim),
            TemporalAttentionModule(feature_dim)
        )
        
        # Part-based feature extraction
        self.part_module = GaitPartitionModule(feature_dim, num_parts=4)
        
        # Temporal modeling with LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim * 4,  # 4 parts
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),  # bidirectional LSTM output
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        # x: (N, C, H, W) where N is number of frames
        N = x.size(0)
        
        # Extract features for each frame
        features = self.backbone(x)  # (N, feature_dim, H', W')
        
        # Part-based features
        part_features = self.part_module(features)  # (N, 4, feature_dim)
        part_features = part_features.view(N, -1)  # (N, 4 * feature_dim)
        
        # Temporal modeling
        # Reshape for LSTM: (1, N, features) - treating sequence as batch
        lstm_input = part_features.unsqueeze(0)
        lstm_out, _ = self.temporal_lstm(lstm_input)
        lstm_out = lstm_out.squeeze(0)  # (N, 1024)
        
        # Final projection
        output = self.projection(lstm_out)  # (N, feature_dim)
        
        return output

class OpenGaitEmbedder:
    def __init__(self, cfg_path=DEFAULT_CFG, weights_path=DEFAULT_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use advanced gait network instead of Plain
        self.model = AdvancedGaitNet(in_channels=1, feature_dim=256)
        self.model.eval()
        self.model.to(self.device)
        
        if os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}: {e}")
                print("Using randomly initialized weights")
        else:
            print(f"Warning: Weights not found at {weights_path}, using randomly initialized weights")
    
    def preprocess_silhouettes(self, silhouettes):
        """Enhanced preprocessing for industrial scenarios"""
        if isinstance(silhouettes, list):
            silhouettes = np.stack(silhouettes, axis=0)
        
        # Normalize to [0, 1]
        if silhouettes.max() > 1:
            silhouettes = silhouettes.astype(np.float32) / 255.0
        
        # Apply morphological operations to handle noise
        from scipy import ndimage
        for i in range(len(silhouettes)):
            # Remove small noise
            silhouettes[i] = ndimage.binary_opening(
                silhouettes[i] > 0.5, 
                structure=np.ones((3, 3))
            ).astype(np.float32)
            
            # Fill small holes
            silhouettes[i] = ndimage.binary_closing(
                silhouettes[i] > 0.5,
                structure=np.ones((5, 5))
            ).astype(np.float32)
        
        # Ensure minimum sequence length for temporal modeling
        min_frames = 16
        if len(silhouettes) < min_frames:
            # Repeat frames to reach minimum
            repeat_factor = min_frames // len(silhouettes) + 1
            silhouettes = np.tile(silhouettes, (repeat_factor, 1, 1))[:min_frames]
        
        return silhouettes

    def extract(self, silhouettes, use_temporal_aggregation=True):
        """
        Extract gait embeddings with advanced temporal modeling
        
        Args:
            silhouettes: numpy array or list of shape (N, H, W), values in [0, 255] or [0, 1]
            use_temporal_aggregation: whether to use temporal aggregation for final embedding
            
        Returns: 
            gait embedding as numpy array
        """
        # Preprocess silhouettes
        silhouettes = self.preprocess_silhouettes(silhouettes)
        
        # Convert to tensor
        silhouettes_tensor = torch.from_numpy(silhouettes).float().unsqueeze(1)  # (N, 1, H, W)
        silhouettes_tensor = silhouettes_tensor.to(self.device)
        
        with torch.no_grad():
            # Extract frame-level features
            frame_features = self.model(silhouettes_tensor)  # (N, feature_dim)
            
            if use_temporal_aggregation:
                # Temporal aggregation strategies
                mean_feat = frame_features.mean(dim=0)
                max_feat, _ = frame_features.max(dim=0)
                std_feat = frame_features.std(dim=0)
                
                # Combine different aggregations
                embedding = torch.cat([mean_feat, max_feat, std_feat], dim=0)
            else:
                # Simple mean pooling
                embedding = frame_features.mean(dim=0)
            
            # L2 normalize for better similarity computation
            embedding = F.normalize(embedding, p=2, dim=0)
            
        return embedding.cpu().numpy()
    
    def extract_sequence_features(self, silhouettes):
        """Extract features for each frame in the sequence"""
        silhouettes = self.preprocess_silhouettes(silhouettes)
        silhouettes_tensor = torch.from_numpy(silhouettes).float().unsqueeze(1)
        silhouettes_tensor = silhouettes_tensor.to(self.device)
        
        with torch.no_grad():
            frame_features = self.model(silhouettes_tensor)
            frame_features = F.normalize(frame_features, p=2, dim=1)
            
        return frame_features.cpu().numpy()

# Usage example:
# embedder = OpenGaitEmbedder()
# embedding = embedder.extract(list_of_silhouette_images)
# sequence_features = embedder.extract_sequence_features(list_of_silhouette_images)