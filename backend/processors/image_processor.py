import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
from torchvision import transforms
import cv2  # This is needed for Grad-CAM

# -----------------------------------------------------------------
# 1. MODEL DEFINITION (Copied from your notebook)
# -----------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1) # Outputs 1 logit for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------
# 2. IMAGE PROCESSOR CLASS (Updated)
# -----------------------------------------------------------------
class ImageProcessor:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- IMPORTANT ---
        # This transform MUST match the validation_tf from your notebook
        # (Normalization was not used in your notebook's val_tf)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # We need to know the class names
        self.class_names = ['fake', 'real'] # 0 = FAKE, 1 = REAL
    
    def load_model(self, model_path: str):
        """Load the trained SimpleCNN model"""
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        print(f"✅ Real SimpleCNN model loaded from {model_path}")
    
    def preprocess_image(self, image_file):
        """Convert uploaded file to tensor"""
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def predict(self, image_tensor):
        """Perform a real prediction with the loaded model"""
        if self.model is None:
            # --- !! UPDATE THIS PATH !! ---
            # Make sure this path is correct for your API's file structure
            self.load_model("D:/hackathonn/researchaton/backend/processors/best_model.pth") 
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor) # Shape (1, 1)
            
            # Apply sigmoid to get a probability (0.0 to 1.0)
            probability = torch.sigmoid(output).item()
        
        # If prob > 0.5, it's class 1 (REAL), else it's class 0 (FAKE)
        predicted_index = 1 if probability > 0.5 else 0
        prediction_label = self.class_names[predicted_index]
        is_fake = (prediction_label == 'fake')
        
        # Calculate confidence
        # If FAKE (prob=0.1), confidence is 1.0 - 0.1 = 0.9
        # If REAL (prob=0.8), confidence is 0.8
        confidence = (1.0 - probability) if is_fake else probability
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'is_fake': is_fake,
            'raw_probability': probability # Good for debugging
        }
    
    def generate_explanations(self, image_tensor):
        """Generate Grad-CAM explanations (Real Implementation)"""
        if self.model is None:
            self.load_model("D:/hackathonn/researchaton/backend/processors/best_model.pth")
            
        self.model.eval()
        
        # We need gradients, so enable grad
        image_tensor = image_tensor.to(self.device).clone().detach().requires_grad_(True)
        
        # --- Grad-CAM Setup ---
        act = []
        grad = []

        def f_hook(module, inp, out):
            act.append(out)

        def b_hook(module, gin, gout):
            grad.append(gout[0])

        # Find last conv layer
        last_conv = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        
        if last_conv is None:
            raise TypeError("Could not find a Conv2d layer for Grad-CAM.")

        h1 = last_conv.register_forward_hook(f_hook)
        h2 = last_conv.register_backward_hook(b_hook)
        # --- End Grad-CAM Setup ---

        out = self.model(image_tensor)
        
        # Backpropagate the output logit
        out.backward()

        # Detach hooks
        h1.remove()
        h2.remove()

        # --- Grad-CAM Calculation ---
        A = act[0][0].cpu().detach().numpy()    # Activations (128, H, W)
        G = grad[0][0].cpu().detach().numpy()   # Gradients (128, H, W)
        
        weights = G.mean(axis=(1, 2)) # (128,)
        cam = np.sum(weights[:, None, None] * A, axis=0) # (H, W)
        cam = np.maximum(cam, 0) # ReLU
        
        # Resize CAM to 224x224
        cam_resized = cv2.resize(cam, (224, 224))
        cam_resized -= np.min(cam_resized)
        if np.max(cam_resized) > 0:
            cam_resized /= np.max(cam_resized) # Normalize to 0-1
        
        # --- Create Overlay ---
        heatmap_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET) # (224, 224, 3) BGR
        
        # Convert base image from (0,1) RGB tensor to (0,255) BGR numpy
        img_array_rgb_uint8 = (image_tensor[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img_array_bgr_uint8 = cv2.cvtColor(img_array_rgb_uint8, cv2.COLOR_RGB2BGR)

        # Create overlay
        overlay = cv2.addWeighted(img_array_bgr_uint8, 0.6, heatmap, 0.4, 0)

        # Convert to base64
        _, buffer = cv2.imencode('.png', overlay)
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'heatmap': f"data:image/png;base64,{overlay_b64}",
            'suspicious_regions': ['(placeholder) face', '(placeholder) eyes'],
            'manipulation_type': '(placeholder) AI-generated'
        }

# Global instance
image_processor = ImageProcessor()