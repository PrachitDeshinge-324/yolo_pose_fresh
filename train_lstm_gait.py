"""
Bidirectional LSTM for Gait-based Person Identification

This script trains a bidirectional LSTM model to identify people based on their gait patterns.
Each person (track_id) is treated as a class, and frame sequences are used as time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import argparse
import os
import json
import joblib
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_best_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using CUDA device: {device_name}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 Using Apple Metal Performance Shaders (MPS)")
        print(f"   PyTorch MPS available: {torch.backends.mps.is_built()}")
        return device
    else:
        device = torch.device('cpu')
        print(f"💻 Using CPU device")
        print(f"   CPU cores: {torch.get_num_threads()}")
        return device

class GaitDataset(Dataset):
    """Dataset class for gait sequences with data augmentation"""
    
    def __init__(self, sequences, labels, sequence_length, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment
        
        # Validate inputs
        assert len(sequences) == len(labels), f"Sequences ({len(sequences)}) and labels ({len(labels)}) must have same length"
        print(f"Dataset created with {len(sequences)} sequences (augment={augment})")
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, sequence):
        """Apply data augmentation to sequence"""
        if not self.augment:
            return sequence
        
        augmented = sequence.copy()
        
        # Add small random noise (5% of std)
        noise_factor = 0.05
        std = np.std(sequence, axis=0, keepdims=True)
        noise = np.random.normal(0, noise_factor * std, sequence.shape)
        augmented += noise
        
        # Random time shift (up to 10% of sequence length)
        shift_amount = np.random.randint(-2, 3)
        if shift_amount != 0:
            augmented = np.roll(augmented, shift_amount, axis=0)
        
        return augmented
    
    def __getitem__(self, idx):
        try:
            sequence = self.sequences[idx]
            sequence = self._augment_sequence(sequence)
            sequence = torch.FloatTensor(sequence)
            
            label_value = self.labels[idx]
            if isinstance(label_value, (list, np.ndarray)):
                label_value = label_value[0] if len(label_value) > 0 else 0
            label = torch.LongTensor([int(label_value)])
            return sequence, label
        except Exception as e:
            print(f"Error in dataset __getitem__ at index {idx}: {e}")
            # Return a dummy sample in case of error
            dummy_sequence = torch.zeros(self.sequence_length, 75)
            dummy_label = torch.LongTensor([0])
            return dummy_sequence, dummy_label

class BidirectionalLSTM(nn.Module):
    """Enhanced Bidirectional LSTM model for gait recognition"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification layers with residual connection
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Input normalization
        x_reshaped = x.view(-1, features)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification with residual connections
        out = self.dropout(context_vector)
        out1 = self.fc1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        
        out = self.fc3(out2)
        
        return out, attention_weights

class GaitTrainer:
    """Main trainer class for gait recognition with enhanced training strategies"""
    
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Initialize metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Set optimal number of workers based on device
        if self.device.type == 'cuda':
            self.num_workers = 4
        elif self.device.type == 'mps':
            self.num_workers = 2
        else:
            self.num_workers = 0
        
        print(f"🔧 DataLoader workers: {self.num_workers}")
        
    def analyze_class_distribution(self, df):
        """Analyze and report class distribution"""
        print("\n=== Class Distribution Analysis ===")
        
        # Track-level analysis
        track_stats = df.groupby('track_id').agg({
            'frame_idx': 'count',
            'person_name': 'first'
        }).reset_index()
        track_stats.columns = ['track_id', 'frame_count', 'person_name']
        
        print("Track statistics:")
        print(track_stats.sort_values('frame_count', ascending=False))
        
        # Check for class imbalance
        min_frames = track_stats['frame_count'].min()
        max_frames = track_stats['frame_count'].max()
        imbalance_ratio = max_frames / min_frames if min_frames > 0 else float('inf')
        
        print(f"\nClass imbalance analysis:")
        print(f"  Min frames per track: {min_frames}")
        print(f"  Max frames per track: {max_frames}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3.0:
            print("⚠️  WARNING: High class imbalance detected! Consider data balancing.")
        
        return track_stats
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the gait data with enhanced validation"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze class distribution
        track_stats = self.analyze_class_distribution(df)
        
        # Filter persons with sufficient data
        min_frames = self.config['min_frames_per_person']
        valid_tracks = track_stats[track_stats['frame_count'] >= min_frames]['track_id'].tolist()
        
        print(f"\nFiltering tracks with at least {min_frames} frames:")
        print(f"Valid tracks: {len(valid_tracks)} out of {len(track_stats)}")
        
        if len(valid_tracks) < 2:
            raise ValueError("Need at least 2 tracks with sufficient data for training!")
        
        # Filter dataframe
        df_filtered = df[df['track_id'].isin(valid_tracks)].copy()
        
        # Select feature columns (exclude metadata)
        feature_columns = [col for col in df_filtered.columns 
                          if col not in ['track_id', 'frame_idx', 'person_name', 'interpolated']]
        
        print(f"Using {len(feature_columns)} features")
        
        # Check for feature quality
        self._analyze_feature_quality(df_filtered, feature_columns)
        
        # Create sequences for each person
        sequences, labels, person_names = self._create_balanced_sequences(df_filtered, feature_columns)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences[0].shape if sequences else 'None'}")
        
        return sequences, labels, person_names, feature_columns
    
    def _analyze_feature_quality(self, df, feature_columns):
        """Analyze feature quality and variability"""
        print("\n=== Feature Quality Analysis ===")
        
        # Calculate feature statistics
        feature_stats = []
        for col in feature_columns:
            values = df[col].values
            stats = {
                'feature': col,
                'mean': np.mean(values),
                'std': np.std(values),
                'variance': np.var(values),
                'min': np.min(values),
                'max': np.max(values),
                'zero_ratio': np.mean(values == 0)
            }
            feature_stats.append(stats)
        
        feature_df = pd.DataFrame(feature_stats)
        
        # Identify low-variance features
        low_variance_threshold = 1e-6
        low_var_features = feature_df[feature_df['variance'] < low_variance_threshold]
        
        print(f"Low variance features ({len(low_var_features)}):")
        if not low_var_features.empty:
            print(low_var_features[['feature', 'variance']].head(10))
        
        # Identify features with high zero ratio
        high_zero_features = feature_df[feature_df['zero_ratio'] > 0.9]
        print(f"High zero-ratio features ({len(high_zero_features)}):")
        if not high_zero_features.empty:
            print(high_zero_features[['feature', 'zero_ratio']].head(10))
        
        return feature_df
    
    def _create_balanced_sequences(self, df, feature_columns):
        """Create balanced sequences from the dataframe"""
        sequences = []
        labels = []
        person_names = []
        
        # Group by person
        grouped = df.groupby('track_id')
        sequence_counts_per_class = []
        
        for track_id, group in grouped:
            # Sort by frame index
            group_sorted = group.sort_values('frame_idx')
            
            # Extract features
            features = group_sorted[feature_columns].values
            
            # Create sequences using sliding window
            sequence_length = self.config['sequence_length']
            step_size = self.config['step_size']
            
            track_sequences = 0
            for i in range(0, len(features) - sequence_length + 1, step_size):
                sequence = features[i:i + sequence_length]
                sequences.append(sequence)
                labels.append(track_id)
                person_names.append(group_sorted.iloc[i]['person_name'])
                track_sequences += 1
            
            sequence_counts_per_class.append(track_sequences)
        
        print(f"Sequences per class: {sequence_counts_per_class}")
        print(f"Min sequences: {min(sequence_counts_per_class)}, Max: {max(sequence_counts_per_class)}")
        
        return sequences, labels, person_names
    
    def prepare_data_loaders(self, sequences, labels):
        """Prepare data loaders with class balancing"""
        print("Preparing data loaders with class balancing...")
        
        # Convert to arrays
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        # Check for class imbalance
        label_counts = Counter(y)
        print(f"Label distribution: {dict(label_counts)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Stratified split to maintain class distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(sss.split(X, y_encoded))
        
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y_encoded[train_val_idx], y_encoded[test_idx]
        
        # Further split train_val into train and validation
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        print(f"Train set: {X_train.shape[0]} sequences")
        print(f"Validation set: {X_val.shape[0]} sequences")
        print(f"Test set: {X_test.shape[0]} sequences")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Save scaler
        scaler_path = os.path.join(self.config['output_dir'], 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Saved feature scaler to {scaler_path}")
        
        # Apply normalization
        X_train_scaled = self._scale_sequences(X_train)
        X_val_scaled = self._scale_sequences(X_val)
        X_test_scaled = self._scale_sequences(X_test)
        
        # Calculate class weights for balanced training
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        # Create sample weights for WeightedRandomSampler
        sample_weights = np.array([class_weights[label] for label in y_train])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"Class weights: {dict(zip(np.unique(y_train), class_weights))}")
        
        # Create datasets with augmentation for training
        train_dataset = GaitDataset(X_train_scaled, y_train, self.config['sequence_length'], augment=True)
        val_dataset = GaitDataset(X_val_scaled, y_val, self.config['sequence_length'], augment=False)
        test_dataset = GaitDataset(X_test_scaled, y_test, self.config['sequence_length'], augment=False)
        
        # Adjust batch size for small datasets
        effective_batch_size = min(self.config['batch_size'], len(train_dataset) // 4)
        if effective_batch_size != self.config['batch_size']:
            print(f"🔧 Adjusted batch size from {self.config['batch_size']} to {effective_batch_size}")
            self.config['batch_size'] = effective_batch_size
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            sampler=sampler,  # Use weighted sampler for balanced training
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False
        )
        
        print(f"🔧 Final batch size: {self.config['batch_size']}")
        print(f"🔧 Train batches: {len(train_loader)} (with weighted sampling)")
        print(f"🔧 Val batches: {len(val_loader)}")
        print(f"🔧 Test batches: {len(test_loader)}")
        
        # Store class weights for loss function
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        return train_loader, val_loader, test_loader, len(np.unique(y_encoded))

    def _scale_sequences(self, sequences):
        """Scale sequence features"""
        scaled_sequences = []
        for seq in sequences:
            seq_reshaped = seq.reshape(-1, seq.shape[-1])
            seq_scaled = self.scaler.transform(seq_reshaped)
            seq_scaled = seq_scaled.reshape(seq.shape)
            scaled_sequences.append(seq_scaled)
        return scaled_sequences
    
    def create_model(self, input_size, num_classes):
        """Create the enhanced bidirectional LSTM model"""
        model = BidirectionalLSTM(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_classes=num_classes,
            dropout=self.config['dropout']
        )
        
        model = model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nEnhanced Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model device: {next(model.parameters()).device}")
        
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch with improved handling"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device, non_blocking=(self.device.type == 'cuda'))
            target = target.squeeze().to(self.device, non_blocking=(self.device.type == 'cuda'))
            
            if target.numel() == 0 or data.size(0) == 0:
                continue
                
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            if data.size(0) != target.size(0):
                min_size = min(data.size(0), target.size(0))
                data = data[:min_size]
                target = target[:min_size]
            
            if data.size(0) == 0:
                continue
            
            optimizer.zero_grad()
            output, attention_weights = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        if total == 0:
            return 0.0, 0.0
        
        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device, non_blocking=(self.device.type == 'cuda'))
                target = target.squeeze().to(self.device, non_blocking=(self.device.type == 'cuda'))
                
                if target.numel() == 0 or data.size(0) == 0:
                    continue
                    
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                
                if data.size(0) != target.size(0):
                    min_size = min(data.size(0), target.size(0))
                    data = data[:min_size]
                    target = target[:min_size]
                
                if data.size(0) == 0:
                    continue
                
                output, attention_weights = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Collect predictions for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        if total == 0:
            return 0.0, 0.0, [], []
        
        epoch_loss = running_loss / max(len(val_loader), 1)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets

    def train_model(self, train_loader, val_loader, input_size, num_classes):
        """Enhanced training loop with better monitoring"""
        print("Starting enhanced training...")
        
        # Create model
        model = self.create_model(input_size, num_classes)
        
        # Weighted loss function for class imbalance
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Adjust learning rate based on device and dataset size
        base_lr = self.config['learning_rate']
        dataset_size_factor = max(0.5, min(2.0, len(train_loader) / 50))  # Scale with dataset size
        
        if self.device.type == 'cuda':
            adjusted_lr = base_lr * 1.2 * dataset_size_factor
        elif self.device.type == 'mps':
            adjusted_lr = base_lr * dataset_size_factor
        else:
            adjusted_lr = base_lr * 0.8 * dataset_size_factor
        
        print(f"📊 Using learning rate: {adjusted_lr:.6f} (base: {base_lr:.6f}, factor: {dataset_size_factor:.2f})")
        
        optimizer = optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_predictions, val_targets = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate based on validation accuracy
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Detailed validation analysis
            if len(val_predictions) > 0 and len(val_targets) > 0:
                unique_preds = np.unique(val_predictions)
                unique_targets = np.unique(val_targets)
                print(f"Val predictions distribution: {dict(zip(*np.unique(val_predictions, return_counts=True)))}")
                print(f"Val targets distribution: {dict(zip(*np.unique(val_targets, return_counts=True)))}")
                
                if len(unique_preds) == 1:
                    print("⚠️  WARNING: Model is predicting only one class!")
            
            # Memory usage info for CUDA
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': self.config,
                    'device': str(self.device),
                    'label_encoder': self.label_encoder,
                    'class_weights': self.class_weights.cpu().numpy()
                }, os.path.join(self.config['output_dir'], 'best_model.pth'))
                patience_counter = 0
                print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return model
    
    def test_model(self, model, test_loader):
        """Test the trained model with detailed analysis"""
        print("Testing model with detailed analysis...")
        
        # Load best model
        checkpoint = torch.load(
            os.path.join(self.config['output_dir'], 'best_model.pth'),
            map_location=self.device,
            weights_only=False  # Allow loading of custom objects like LabelEncoder
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_attention_weights = []
        all_confidences = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=(self.device.type == 'cuda'))
                target = target.squeeze().to(self.device, non_blocking=(self.device.type == 'cuda'))
                
                if target.numel() == 0 or data.size(0) == 0:
                    continue
                    
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                
                if data.size(0) != target.size(0):
                    min_size = min(data.size(0), target.size(0))
                    data = data[:min_size]
                    target = target[:min_size]
                
                if data.size(0) == 0:
                    continue
                
                output, attention_weights = model(data)
                
                # Get probabilities and confidence
                probabilities = torch.softmax(output, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                predicted_np = predicted.cpu().numpy()
                target_np = target.cpu().numpy()
                confidences_np = confidences.cpu().numpy()
                
                if predicted_np.ndim == 0:
                    predicted_np = [predicted_np.item()]
                if target_np.ndim == 0:
                    target_np = [target_np.item()]
                if confidences_np.ndim == 0:
                    confidences_np = [confidences_np.item()]
                
                all_predictions.extend(predicted_np)
                all_targets.extend(target_np)
                all_confidences.extend(confidences_np)
                all_attention_weights.append(attention_weights.cpu().numpy())
        
        if len(all_targets) == 0:
            print("No test samples were processed!")
            return [], [], [], [], 0.0
        
        # Calculate metrics
        test_acc = accuracy_score(all_targets, all_predictions)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Average Confidence: {np.mean(all_confidences):.4f}")
        print(f"Confidence Std: {np.std(all_confidences):.4f}")
        
        # Detailed prediction analysis
        prediction_counts = Counter(all_predictions)
        target_counts = Counter(all_targets)
        
        print(f"\nPrediction distribution: {dict(prediction_counts)}")
        print(f"Target distribution: {dict(target_counts)}")
        
        # Check if model is predicting only one class
        if len(prediction_counts) == 1:
            print("⚠️  CRITICAL: Model is predicting only ONE class!")
            pred_class = list(prediction_counts.keys())[0]
            print(f"   Always predicting class: {pred_class}")
            print(f"   Original track_id: {self.label_encoder.inverse_transform([pred_class])[0]}")
        
        print("\nClassification Report:")
        class_names = [f"Person_{self.label_encoder.inverse_transform([i])[0]}" 
                      for i in range(len(self.label_encoder.classes_))]
        
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        return all_predictions, all_targets, all_attention_weights, all_confidences, test_acc

    def visualize_results(self, predictions, targets, attention_weights, confidences):
        """Create comprehensive visualization plots"""
        print("Creating comprehensive visualizations...")
        
        if len(targets) == 0 or len(predictions) == 0:
            print("No valid test results to visualize!")
            return
        
        # Create main figure
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Training curves
        plt.subplot(3, 4, 1)
        if self.train_losses and self.val_losses:
            plt.plot(self.train_losses, label='Train Loss', linewidth=2)
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
            plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        if self.train_accuracies and self.val_accuracies:
            plt.plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
            plt.plot(self.val_accuracies, label='Validation Accuracy', linewidth=2)
            plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        plt.subplot(3, 4, 3)
        try:
            cm = confusion_matrix(targets, predictions)
            class_names = [f"P_{self.label_encoder.inverse_transform([i])[0]}" 
                          for i in range(len(self.label_encoder.classes_))]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            plt.title('Confusion Matrix')
        
        # 3. Prediction Distribution
        plt.subplot(3, 4, 4)
        prediction_counts = Counter(predictions)
        target_counts = Counter(targets)
        
        classes = list(range(len(self.label_encoder.classes_)))
        pred_counts = [prediction_counts.get(c, 0) for c in classes]
        target_counts_list = [target_counts.get(c, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, target_counts_list, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Prediction vs True Distribution')
        plt.legend()
        plt.xticks(x, [f"P_{self.label_encoder.inverse_transform([i])[0]}" for i in classes], rotation=45)
        
        # 4. Confidence Distribution
        plt.subplot(3, 4, 5)
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Per-class accuracy
        plt.subplot(3, 4, 6)
        class_names = [f"P_{self.label_encoder.inverse_transform([i])[0]}" 
                      for i in range(len(self.label_encoder.classes_))]
        class_accuracies = []
        for i in range(len(class_names)):
            mask = np.array(targets) == i
            if mask.sum() > 0:
                acc = (np.array(predictions)[mask] == i).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)
        
        bars = plt.bar(class_names, class_accuracies, color=sns.color_palette("husl", len(class_names)))
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Person')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}', ha='center', va='bottom')
        
        # 6. Attention weights visualization
        plt.subplot(3, 4, 7)
        if attention_weights and len(attention_weights) > 0:
            avg_attention = np.mean(np.concatenate(attention_weights, axis=0), axis=0)
            plt.plot(avg_attention.squeeze(), linewidth=2)
            plt.title('Average Attention Weights Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
        
        # 7. Class-wise confidence
        plt.subplot(3, 4, 8)
        class_confidences = defaultdict(list)
        for pred, conf in zip(predictions, confidences):
            class_confidences[pred].append(conf)
        
        class_conf_means = []
        class_conf_stds = []
        valid_classes = []
        
        for i in range(len(self.label_encoder.classes_)):
            if i in class_confidences:
                confs = class_confidences[i]
                class_conf_means.append(np.mean(confs))
                class_conf_stds.append(np.std(confs))
                valid_classes.append(f"P_{self.label_encoder.inverse_transform([i])[0]}")
        
        if valid_classes:
            plt.errorbar(range(len(valid_classes)), class_conf_means, yerr=class_conf_stds, 
                        fmt='o-', capsize=5, capthick=2)
            plt.xlabel('Class')
            plt.ylabel('Confidence')
            plt.title('Per-Class Confidence')
            plt.xticks(range(len(valid_classes)), valid_classes, rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 8. Model performance summary
        plt.subplot(3, 4, 9)
        plt.axis('off')
        
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        test_acc = accuracy_score(targets, predictions) if len(targets) > 0 else 0
        
        summary_text = f"""
        Enhanced Model Performance Summary
        
        Final Training Accuracy: {final_train_acc:.2f}%
        Final Validation Accuracy: {final_val_acc:.2f}%
        Test Accuracy: {test_acc:.2f}%
        
        Average Confidence: {np.mean(confidences):.3f}
        Confidence Std: {np.std(confidences):.3f}
        
        Number of Classes: {len(self.label_encoder.classes_)}
        Total Test Samples: {len(targets)}
        
        Model Configuration:
        - Sequence Length: {self.config['sequence_length']}
        - Hidden Size: {self.config['hidden_size']}
        - Layers: {self.config['num_layers']}
        - Dropout: {self.config['dropout']}
        
        Training Device: {self.device.type.upper()}
        Class Balancing: Enabled
        Data Augmentation: Enabled
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'enhanced_training_results.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved enhanced training results plot")
        plt.show()
        
    def save_results(self, predictions, targets, confidences, test_acc):
        """Save comprehensive training results"""
        results = {
            'test_accuracy': float(test_acc),
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config,
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'device_used': str(self.device),
            'class_weights': self.class_weights.cpu().numpy().tolist()
        }
        
        # Save results
        import pickle
        with open(os.path.join(self.config['output_dir'], 'enhanced_training_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save config as JSON
        config_with_device = self.config.copy()
        config_with_device['device_used'] = str(self.device)
        with open(os.path.join(self.config['output_dir'], 'config.json'), 'w') as f:
            json.dump(config_with_device, f, indent=2)
        
        print(f"Enhanced results saved to {self.config['output_dir']}")

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Bidirectional LSTM for Gait Recognition')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with gait features')
    parser.add_argument('--output_dir', type=str, default='lstm_results', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of input sequences')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for sliding window')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--min_frames_per_person', type=int, default=40, help='Minimum frames per person')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'sequence_length': args.sequence_length,
        'step_size': args.step_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'min_frames_per_person': args.min_frames_per_person,
        'patience': args.patience,
        'output_dir': args.output_dir
    }
    
    print("=== Enhanced Bidirectional LSTM Gait Recognition Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize trainer
    trainer = GaitTrainer(config)
    
    # Load and preprocess data
    sequences, labels, person_names, feature_columns = trainer.load_and_preprocess_data(args.data)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader, num_classes = trainer.prepare_data_loaders(sequences, labels)
    
    print(f"Number of features: {len(feature_columns)}")
    print(f"Number of classes: {num_classes}")
    
    # Train model
    model = trainer.train_model(train_loader, val_loader, len(feature_columns), num_classes)
    
    # Test model
    predictions, targets, attention_weights, confidences, test_acc = trainer.test_model(model, test_loader)
    
    # Create visualizations
    trainer.visualize_results(predictions, targets, attention_weights, confidences)
    
    # Save results
    trainer.save_results(predictions, targets, confidences, test_acc)
    
    print(f"\n=== Enhanced Training Complete ===")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Average Confidence: {np.mean(confidences):.4f}")
    print(f"Training Device: {trainer.device}")
    print(f"Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()