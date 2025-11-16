import torch
import torch.nn.functional as F
import numpy as np

from models import TextDetector

class TextProcessor:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_path: str):
        """Load trained model - placeholder"""
        self.model = TextDetector()
        self.model.eval()
        self.model.to(self.device)
        print("✅ Text model loaded (placeholder)")
    
    def preprocess_text(self, text: str):
        """Simple text preprocessing - replace with proper tokenization"""
        # Placeholder: convert text to token indices
        words = text.lower().split()[:50]  # Limit to 50 words
        tokens = [hash(word) % 1000 for word in words]  # Simple hash tokenization
        if len(tokens) < 50:
            tokens.extend([0] * (50 - len(tokens)))  # Pad to 50
        return torch.tensor([tokens], dtype=torch.long)
    
    def predict(self, text_tensor):
        """Text prediction with placeholder logic"""
        if self.model is None:
            self.load_model("models/text_detector.pth")
        
        text_tensor = text_tensor.to(self.device)
        
        # Placeholder inference
        with torch.no_grad():
            outputs = torch.randn(1, 2)  # Replace with self.model(text_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction_idx = torch.max(probabilities, 1)
        
        is_fake = (prediction_idx.item() == 1)
        prediction_label = "AI-generated" if is_fake else "Human-written"
        
        return {
            'prediction': prediction_label,
            'confidence': confidence.item(),
            'is_fake': is_fake
        }
    
    def generate_explanations(self, text: str, text_tensor):
        """Generate SHAP-like explanations (placeholder)"""
        words = text.split()[:10]  # First 10 words
        word_scores = np.random.rand(len(words)).tolist()
        
        return {
            'word_importance': dict(zip(words, word_scores)),
            'key_phrases': ['repetitive patterns', 'unnatural flow'],
            'detection_reasons': ['Low perplexity', 'Pattern repetition']
        }

# Global instance
text_processor = TextProcessor()