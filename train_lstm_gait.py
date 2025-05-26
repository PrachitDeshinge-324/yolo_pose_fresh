"""
Bidirectional LSTM for Gait-based Person Identification

This script trains a bidirectional LSTM model to identify people based on their gait patterns.
Each person (track_id) is treated as a class, and frame sequences are used as time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from collections import defaultdict
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
    """Dataset class for gait sequences with better error handling"""
    
    def __init__(self, sequences, labels, sequence_length):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        
        # 🔥 Validate inputs
        assert len(sequences) == len(labels), f"Sequences ({len(sequences)}) and labels ({len(labels)}) must have same length"
        print(f"Dataset created with {len(sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            sequence = torch.FloatTensor(self.sequences[idx])
            # 🔥 Ensure label is a single integer, not a list
            label_value = self.labels[idx]
            if isinstance(label_value, (list, np.ndarray)):
                label_value = label_value[0] if len(label_value) > 0 else 0
            label = torch.LongTensor([int(label_value)])
            return sequence, label
        except Exception as e:
            print(f"Error in dataset __getitem__ at index {idx}: {e}")
            # Return a dummy sample in case of error
            dummy_sequence = torch.zeros(self.sequence_length, 75)  # 75 features
            dummy_label = torch.LongTensor([0])
            return dummy_sequence, dummy_label
        
class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM model for gait recognition"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
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
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        out = self.dropout(context_vector)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights

class GaitTrainer:
    """Main trainer class for gait recognition"""
    
    def __init__(self, config):
        self.config = config
        # 🔥 Use the best available device
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
            self.num_workers = 4  # More workers for CUDA
        elif self.device.type == 'mps':
            self.num_workers = 2  # Moderate for MPS
        else:
            self.num_workers = 0  # Single thread for CPU
        
        print(f"🔧 DataLoader workers: {self.num_workers}")
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the gait data"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Get unique persons and their statistics
        person_stats = df.groupby('track_id').agg({
            'frame_idx': ['count', 'min', 'max'],
            'person_name': 'first'
        }).reset_index()
        
        person_stats.columns = ['track_id', 'frame_count', 'min_frame', 'max_frame', 'person_name']
        print(f"\nPerson statistics:")
        print(person_stats)
        
        # Filter persons with sufficient data
        min_frames = self.config['min_frames_per_person']
        valid_persons = person_stats[person_stats['frame_count'] >= min_frames]['track_id'].tolist()
        
        print(f"\nFiltering persons with at least {min_frames} frames:")
        print(f"Valid persons: {len(valid_persons)} out of {len(person_stats)}")
        
        if len(valid_persons) < 2:
            raise ValueError("Need at least 2 persons with sufficient data for training!")
        
        # Filter dataframe
        df_filtered = df[df['track_id'].isin(valid_persons)].copy()
        
        # Select feature columns (exclude metadata)
        feature_columns = [col for col in df_filtered.columns 
                          if col not in ['track_id', 'frame_idx', 'person_name', 'interpolated']]
        
        print(f"Using {len(feature_columns)} features: {feature_columns[:10]}...")
        
        # Create sequences for each person
        sequences, labels, person_names = self._create_sequences(df_filtered, feature_columns)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences[0].shape if sequences else 'None'}")
        
        return sequences, labels, person_names, feature_columns
    
    def _create_sequences(self, df, feature_columns):
        """Create sequences from the dataframe"""
        sequences = []
        labels = []
        person_names = []
        
        # Group by person
        grouped = df.groupby('track_id')
        
        for track_id, group in grouped:
            # Sort by frame index
            group_sorted = group.sort_values('frame_idx')
            
            # Extract features
            features = group_sorted[feature_columns].values
            
            # Create sequences using sliding window
            sequence_length = self.config['sequence_length']
            step_size = self.config['step_size']
            
            for i in range(0, len(features) - sequence_length + 1, step_size):
                sequence = features[i:i + sequence_length]
                sequences.append(sequence)
                labels.append(track_id)
                person_names.append(group_sorted.iloc[i]['person_name'])
        
        return sequences, labels, person_names
    
    def prepare_data_loaders(self, sequences, labels):
        """Prepare data loaders with better batch handling"""
        print("Preparing data loaders...")
        
        # Convert to arrays
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data: 60% train, 20% val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
        )
        
        print(f"Train set: {X_train.shape[0]} sequences")
        print(f"Validation set: {X_val.shape[0]} sequences")
        print(f"Test set: {X_test.shape[0]} sequences")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Apply normalization
        X_train_scaled = self._scale_sequences(X_train)
        X_val_scaled = self._scale_sequences(X_val)
        X_test_scaled = self._scale_sequences(X_test)
        
        # Create datasets
        train_dataset = GaitDataset(X_train_scaled, y_train, self.config['sequence_length'])
        val_dataset = GaitDataset(X_val_scaled, y_val, self.config['sequence_length'])
        test_dataset = GaitDataset(X_test_scaled, y_test, self.config['sequence_length'])
        
        # 🔥 Better batch size handling for small datasets
        effective_batch_size = min(self.config['batch_size'], len(train_dataset), len(val_dataset))
        if effective_batch_size != self.config['batch_size']:
            print(f"🔧 Adjusted batch size from {self.config['batch_size']} to {effective_batch_size}")
            self.config['batch_size'] = effective_batch_size
        
        # Create data loaders with drop_last=False to include all samples
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False  # 🔥 Don't drop last incomplete batch
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False  # 🔥 Don't drop last incomplete batch
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False  # 🔥 Don't drop last incomplete batch
        )
        
        print(f"🔧 Final batch size: {self.config['batch_size']}")
        print(f"🔧 Train batches: {len(train_loader)}")
        print(f"🔧 Val batches: {len(val_loader)}")
        print(f"🔧 Test batches: {len(test_loader)}")
        
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
        """Create the bidirectional LSTM model"""
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
        
        print(f"\nModel Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model device: {next(model.parameters()).device}")
        print(model)
        
        return model
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch with batch size handling"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data = data.to(self.device, non_blocking=(self.device.type == 'cuda'))
                target = target.squeeze().to(self.device, non_blocking=(self.device.type == 'cuda'))
                
                # 🔥 Skip empty batches (fixes the batch size mismatch)
                if target.numel() == 0 or data.size(0) == 0:
                    continue
                    
                # 🔥 Ensure target is 1D and matches batch size
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                
                # 🔥 Check batch size consistency
                if data.size(0) != target.size(0):
                    print(f"Warning: Batch size mismatch - data: {data.size(0)}, target: {target.size(0)}")
                    min_size = min(data.size(0), target.size(0))
                    data = data[:min_size]
                    target = target[:min_size]
                
                if data.size(0) == 0:  # Skip if still empty after adjustment
                    continue
                
                output, attention_weights = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Handle case where no valid batches were processed
        if total == 0:
            return 0.0, 0.0
        
        epoch_loss = running_loss / max(len(val_loader), 1)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch with batch size handling"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device, non_blocking=(self.device.type == 'cuda'))
            target = target.squeeze().to(self.device, non_blocking=(self.device.type == 'cuda'))
            
            # 🔥 Skip empty batches
            if target.numel() == 0 or data.size(0) == 0:
                continue
                
            # 🔥 Ensure target is 1D and matches batch size
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            # 🔥 Check batch size consistency
            if data.size(0) != target.size(0):
                print(f"Warning: Batch size mismatch - data: {data.size(0)}, target: {target.size(0)}")
                min_size = min(data.size(0), target.size(0))
                data = data[:min_size]
                target = target[:min_size]
            
            if data.size(0) == 0:  # Skip if still empty after adjustment
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
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Handle case where no valid batches were processed
        if total == 0:
            return 0.0, 0.0
        
        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def train_model(self, train_loader, val_loader, input_size, num_classes):
        """Main training loop"""
        print("Starting training...")
        
        # Create model
        model = self.create_model(input_size, num_classes)
        
        # Loss and optimizer with device-specific optimizations
        criterion = nn.CrossEntropyLoss()
        
        # Adjust learning rate based on device
        base_lr = self.config['learning_rate']
        if self.device.type == 'cuda':
            # Can use higher learning rate with CUDA
            adjusted_lr = base_lr * 1.2
        elif self.device.type == 'mps':
            # Moderate learning rate for MPS
            adjusted_lr = base_lr
        else:
            # Lower learning rate for CPU
            adjusted_lr = base_lr * 0.8
        
        print(f"📊 Using learning rate: {adjusted_lr:.6f} (base: {base_lr:.6f})")
        
        optimizer = optim.Adam(model.parameters(), lr=adjusted_lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        # Device-specific memory optimization
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
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
                    'device': str(self.device)
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
        """Test the trained model"""
        print("Testing model...")
        
        # Load best model with device mapping
        checkpoint = torch.load(
            os.path.join(self.config['output_dir'], 'best_model.pth'),
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_attention_weights = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=(self.device.type == 'cuda'))
                target = target.squeeze().to(self.device, non_blocking=(self.device.type == 'cuda'))
                
                output, attention_weights = model(data)
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_attention_weights.append(attention_weights.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(all_targets, all_predictions)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        
        # Get class names
        class_names = [f"Person_{self.label_encoder.inverse_transform([i])[0]}" 
                      for i in range(len(self.label_encoder.classes_))]
        
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        return all_predictions, all_targets, all_attention_weights, test_acc
    
    def visualize_results(self, predictions, targets, attention_weights):
        """Create visualization plots"""
        print("Creating visualizations...")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training curves
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
        plt.plot(self.val_accuracies, label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(targets, predictions)
        class_names = [f"P_{self.label_encoder.inverse_transform([i])[0]}" 
                      for i in range(len(self.label_encoder.classes_))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 3. Class-wise accuracy
        plt.subplot(2, 3, 4)
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
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}', ha='center', va='bottom')
        
        # 4. Attention weights visualization
        plt.subplot(2, 3, 5)
        if attention_weights:
            # Average attention weights across all samples
            avg_attention = np.mean(np.concatenate(attention_weights, axis=0), axis=0)
            plt.plot(avg_attention.squeeze(), linewidth=2)
            plt.title('Average Attention Weights Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
        
        # 5. Model performance summary with device info
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Calculate summary statistics
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        test_acc = accuracy_score(targets, predictions)
        
        summary_text = f"""
        Model Performance Summary
        
        Final Training Accuracy: {final_train_acc:.2f}%
        Final Validation Accuracy: {final_val_acc:.2f}%
        Test Accuracy: {test_acc:.2f}%
        
        Number of Classes: {len(class_names)}
        Total Test Samples: {len(targets)}
        
        Model Configuration:
        - Sequence Length: {self.config['sequence_length']}
        - Hidden Size: {self.config['hidden_size']}
        - Number of Layers: {self.config['num_layers']}
        - Dropout: {self.config['dropout']}
        
        Training Device: {self.device.type.upper()}
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.config['output_dir'], 'training_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed plots
        self._create_detailed_plots(predictions, targets, attention_weights)
    
    def _create_detailed_plots(self, predictions, targets, attention_weights):
        """Create additional detailed visualization plots"""
        
        # Per-class confusion matrix (normalized)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        cm = confusion_matrix(targets, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        class_names = [f"P_{self.label_encoder.inverse_transform([i])[0]}" 
                      for i in range(len(self.label_encoder.classes_))]
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Raw Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Sample attention weights for different classes
        if attention_weights:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            all_attention = np.concatenate(attention_weights, axis=0)
            unique_targets = np.unique(targets)
            
            for i, target_class in enumerate(unique_targets[:4]):  # Show first 4 classes
                if i < len(axes):
                    # Find samples of this class
                    class_mask = np.array(targets) == target_class
                    class_indices = np.where(class_mask)[0]
                    
                    if len(class_indices) > 0:
                        # Get attention weights for this class
                        sample_idx = class_indices[0]  # Take first sample
                        attention_seq = all_attention[sample_idx].squeeze()
                        
                        axes[i].plot(attention_seq, linewidth=2)
                        axes[i].set_title(f'Attention Weights - Person {self.label_encoder.inverse_transform([target_class])[0]}')
                        axes[i].set_xlabel('Time Step')
                        axes[i].set_ylabel('Attention Weight')
                        axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output_dir'], 'attention_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self, predictions, targets, test_acc):
        """Save training results and model info"""
        results = {
            'test_accuracy': float(test_acc),
            'predictions': predictions,
            'targets': targets,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config,
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'device_used': str(self.device)
        }
        
        # Save results
        import pickle
        with open(os.path.join(self.config['output_dir'], 'training_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save config as JSON
        config_with_device = self.config.copy()
        config_with_device['device_used'] = str(self.device)
        with open(os.path.join(self.config['output_dir'], 'config.json'), 'w') as f:
            json.dump(config_with_device, f, indent=2)
        
        print(f"Results saved to {self.config['output_dir']}")

def main():
    parser = argparse.ArgumentParser(description='Train Bidirectional LSTM for Gait Recognition')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with gait features')
    parser.add_argument('--output_dir', type=str, default='lstm_results', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=30, help='Length of input sequences')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for sliding window')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--min_frames_per_person', type=int, default=50, help='Minimum frames per person')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
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
    
    print("=== Bidirectional LSTM Gait Recognition Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize trainer (automatically detects best device)
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
    predictions, targets, attention_weights, test_acc = trainer.test_model(model, test_loader)
    
    # Create visualizations
    trainer.visualize_results(predictions, targets, attention_weights)
    
    # Save results
    trainer.save_results(predictions, targets, test_acc)
    
    print(f"\n=== Training Complete ===")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Training Device: {trainer.device}")
    print(f"Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()