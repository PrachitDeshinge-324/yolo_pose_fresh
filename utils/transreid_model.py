import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from PIL import Image
import os
from functools import lru_cache
import cv2

def get_best_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def interpolate_pos_embed_rectangular(model, state_dict):
    """
    Interpolate position embeddings from checkpoint to model when grids don't match,
    properly handling rectangular (non-square) grid layouts.
    """
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        
        # Get grid sizes from model
        model_grid_size = (model.patch_embed.num_y, model.patch_embed.num_x)
        model_num_patches = model_grid_size[0] * model_grid_size[1]
        
        # Get grid size from checkpoint
        checkpoint_num_patches = pos_embed_checkpoint.shape[1] - 1  # minus cls token
        
        # Estimate checkpoint grid size (try to get close to original aspect ratio)
        # For TransReID, most common is 24x10 grid (240 patches) or 24×9 (216 patches)
        if checkpoint_num_patches in [210, 240, 216]:
            if checkpoint_num_patches == 210:
                checkpoint_grid_size = (21, 10)  # 21×10=210 patches
            elif checkpoint_num_patches == 240:
                checkpoint_grid_size = (24, 10)  # 24×10=240 patches
            elif checkpoint_num_patches == 216:
                checkpoint_grid_size = (24, 9)   # 24×9=216 patches
        else:
            # For other sizes, try to approximate proportionally
            ratio = model_grid_size[0] / model_grid_size[1]  # height/width ratio
            checkpoint_h = int((checkpoint_num_patches * ratio) ** 0.5)
            checkpoint_w = checkpoint_num_patches // checkpoint_h
            while checkpoint_h * checkpoint_w != checkpoint_num_patches:
                checkpoint_h -= 1
                checkpoint_w = checkpoint_num_patches // checkpoint_h
            checkpoint_grid_size = (checkpoint_h, checkpoint_w)
            
        print(f"Interpolating position embeddings:")
        print(f"  - Checkpoint grid: {checkpoint_grid_size[0]}×{checkpoint_grid_size[1]} ({checkpoint_num_patches} patches)")
        print(f"  - Model grid: {model_grid_size[0]}×{model_grid_size[1]} ({model_num_patches} patches)")
        
        # Handle class token and reshape
        cls_pos_embed = pos_embed_checkpoint[:, 0:1, :]
        pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]
        
        # Reshape into grid
        pos_embed_checkpoint = pos_embed_checkpoint.reshape(
            1, checkpoint_grid_size[0], checkpoint_grid_size[1], embedding_size
        ).permute(0, 3, 1, 2)
        
        # Interpolate to new size
        import torch.nn.functional as F
        pos_embed_new = F.interpolate(
            pos_embed_checkpoint, 
            size=model_grid_size, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Reshape back
        pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(
            1, model_grid_size[0] * model_grid_size[1], embedding_size
        )
        
        # Attach class token
        new_pos_embed = torch.cat((cls_pos_embed, pos_embed_new), dim=1)
        state_dict['pos_embed'] = new_pos_embed
        
    return state_dict

class TransReIDModel:
    def __init__(self, weights_path, device=None):
        """
        Initialize TransReID model for person re-identification
        
        Args:
            weights_path: Path to pre-trained weights
            device: Device to run the model on
        """
        self.device = device if device is not None else get_best_device()
        
        # TransReID feature dimension (commonly 768 for ViT-base)
        self.feature_dim = 768
        
        # Initialize model
        self.model = self._load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, weights_path):
        if not os.path.exists(weights_path):
            print(f"Warning: TransReID weights file not found at {weights_path}")
            raise FileNotFoundError(f"TransReID weights file not found at {weights_path}")

        try:
            print(f"Loading TransReID weights from {weights_path}...")
            model = self._create_transreid_model()
            # Load the state dict and remove 'base.' prefix if present
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove 'base.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('base.'):
                    new_state_dict[k[5:]] = v
                else:
                    new_state_dict[k] = v
            
            # Apply position embedding interpolation
            new_state_dict = interpolate_pos_embed_rectangular(model, new_state_dict)
            
            # Load state dict
            model.load_state_dict(new_state_dict, strict=False)
            print("TransReID weights loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise
    
    def _create_transreid_model(self):
        model = vit_base_patch16_224_TransReID(
            img_size=(256, 128),   # or your dataset's image size
            stride_size=16,
            drop_path_rate=0.1,
            camera=0,
            view=0,
            local_feature=False,
            sie_xishu=1.5
        )
        return model
    
    def extract_features(self, person_crop):
        """
        Extract feature embeddings from a person crop
        
        Args:
            person_crop: Cropped image of a person (BGR format)
        
        Returns:
            Feature embedding tensor
        """
        # Convert BGR to RGB
        if person_crop.size == 0:
            return torch.zeros(self.feature_dim).to(self.device)
            
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # Preprocess
        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Normalize feature vector
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        
        return normalized_features.squeeze()

# add cached factory
@lru_cache(maxsize=None)
def load_transreid_model(weights_path: str, device: str):
    """
    Return a cached TransReIDModel instance for given weights and device.
    """
    return TransReIDModel(weights_path, device)