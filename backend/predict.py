import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import cv2
import numpy as np
import base64
import types
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import VisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer()

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_model", "deepfake_vit.pth")
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except RuntimeError as e:
        print(f"Warning: Incompatible model weights found at {model_path}. Please retrain the model. Error: {e}")
else:
    print(f"Warning: Model weights not found at {model_path}. Using uninitialized weights.")

model.to(device)
model.eval()

# =====================================================================
# ADVANCED FEATURE: PYTORCH VISION TRANSFORMER MONKEY-PATCH
# =====================================================================
# By default, the PyTorch `torchvision` ViT hides its internal self-attention 
# weights to save VRAM memory during inference. However, we need these weights 
# to show the user exactly what the AI is looking at (Stage 02 Heatmap).
# 
# We achieve this by "Monkey-Patching" (overwriting) the forward pass of the 
# very last Transformer block. We intercept the MultiheadAttention call and 
# force `need_weights=True`, allowing us to extract the true mathematical 
# attention map without breaking the model!
attention_maps = []
def new_forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    x, weights = self.self_attention(x, x, x, need_weights=True)
    attention_maps.append(weights)
    x = self.dropout(x)
    x = x + input
    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y

model.vit.encoder.layers[-1].forward = types.MethodType(new_forward, model.vit.encoder.layers[-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# MAIN PREDICTION FUNCTION
# =====================================================================
# This function is called by the Flask API whenever a user uploads an image.
# It processes the image, runs it through the ViT, and generates the heatmap.
def predict_image(img_path):
    global attention_maps
    attention_maps.clear() # Clear the old attention map from the previous run
    
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return "ERROR", 0.0, None, None, 0.0

    image_tensor = transform(image).unsqueeze(0).to(device)

    embedding_b64 = None
    
    start_time = time.time()
    with torch.no_grad():
        # =====================================================================
        # PATCH EMBEDDING EXTRACTION (STAGE 01.5)
        # =====================================================================
        # Extract the initial 768-dimensional patch embeddings
        patch_embeddings = model.vit.conv_proj(image_tensor)
        
        # Calculate the magnitude (mean intensity) across all 768 dimensions
        embedding_map = patch_embeddings.mean(dim=1).squeeze(0).cpu().numpy()
        
        # Normalize and convert to 8-bit image
        embedding_map = (embedding_map - embedding_map.min()) / (embedding_map.max() - embedding_map.min() + 1e-8)
        embedding_map = np.uint8(255 * embedding_map)
        
        # Resize using INTER_NEAREST to preserve the 16x16 blocky patch structure
        embedding_map = cv2.resize(embedding_map, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        orig_img_bgr = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
        
        # Apply VIRIDIS colormap to distinguish from the JET attention map
        embedding_heatmap = cv2.applyColorMap(embedding_map, cv2.COLORMAP_VIRIDIS)
        
        # Blend and encode
        blended_emb = cv2.addWeighted(orig_img_bgr, 0.4, embedding_heatmap, 0.6, 0)
        _, buffer_emb = cv2.imencode('.jpg', blended_emb)
        embedding_b64 = base64.b64encode(buffer_emb).decode('utf-8')

        output, _ = model(image_tensor)
        
    end_time = time.time()
    inference_time = round(end_time - start_time, 3)

    probs = torch.softmax(output, dim=1)
    confidence = probs.max().item() * 100
    prediction = torch.argmax(output, dim=1)
    result = "FAKE" if prediction.item() == 0 else "REAL"
    
    # =====================================================================
    # HEATMAP GENERATION (STAGE 02 VISUALIZATION)
    # =====================================================================
    # Here we take the raw attention tensor extracted from the monkey-patch,
    # and convert it into a visual heatmap overlay using OpenCV.
    heatmap_b64 = None
    if len(attention_maps) > 0:
        attn = attention_maps[0][0] # Get the attention tensor (Shape: [197, 197])
        
        # We only care about the [CLS] token (index 0) and how it attends to 
        # the 196 spatial image patches (index 1 to 196).
        # We extract this vector and reshape it into a 14x14 grid.
        cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
        
        # Normalize the tensor values between 0 and 1
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        
        # Convert to 8-bit image format (0-255) and resize to match the 224x224 image
        cls_attn = np.uint8(255 * cls_attn)
        cls_attn = cv2.resize(cls_attn, (224, 224))
        
        orig_img = np.array(image.resize((224, 224)))
        # OpenCV uses BGR color format, but PIL uses RGB. We must convert it.
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        # Apply the JET colormap (Blue=Low Attention, Red=High Attention)
        heatmap = cv2.applyColorMap(cls_attn, cv2.COLORMAP_JET)
        
        # Blend the original image and the heatmap together (50% opacity each)
        blended = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
        
        # Convert back to Base64
        _, buffer = cv2.imencode('.jpg', blended)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

    return result, round(confidence, 2), heatmap_b64, embedding_b64, inference_time