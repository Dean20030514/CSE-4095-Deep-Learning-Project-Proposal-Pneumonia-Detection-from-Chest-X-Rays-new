import argparse
from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DISCLAIMER = "⚠️ For research/education only. Not a medical device. Do not use for clinical diagnosis."


def load_model(ckpt_path: Path):
    """Load model from checkpoint with config"""
    from src.models.factory import build_model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    
    num_classes = len(ckpt['classes'])
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    idx_to_class = {v: k for k, v in ckpt['classes'].items()}
    class_to_idx = ckpt['classes']
    
    return model, idx_to_class, class_to_idx, img_size, model_name


def generate_gradcam(model, image_tensor, target_class, model_name):
    """Generate Grad-CAM heatmap"""
    from src.utils.gradcam import GradCAM
    
    # Determine target layer based on model
    if 'resnet' in model_name.lower():
        target_layer = 'layer4'
    elif 'efficientnet' in model_name.lower():
        target_layer = 'features'
    else:
        target_layer = 'layer4'  # default
    
    try:
        gradcam = GradCAM(model, target_layer)
        model.eval()
        image_tensor.requires_grad = True
        logits = model(image_tensor)
        cam = gradcam(logits, target_class)
        return cam.cpu().numpy()
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    # Resize heatmap to match image
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.BILINEAR))
    
    # Normalize heatmap
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Apply colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    
    # Convert image to array
    img_array = np.array(image).astype(float) / 255.0
    
    # Ensure same dimensions
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Blend
    overlayed = (1 - alpha) * img_array + alpha * heatmap_colored
    overlayed = (overlayed * 255).astype(np.uint8)
    
    return Image.fromarray(overlayed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='runs/best.pt')
    parser.add_argument('--demo_mode', action='store_true')
    args, unknown = parser.parse_known_args()

    st.set_page_config(page_title="Pneumonia X-ray Detection", layout='wide')
    st.title("🫁 Pneumonia X-ray Detection System")
    st.markdown(f"**{DISCLAIMER}**")

    # Demo mode
    if args.demo_mode or not Path(args.ckpt).exists():
        st.warning("⚙️ Demo mode: no model weights provided. Showing static interface.")
        uploaded = st.file_uploader("Upload chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"]) 
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Input image", use_container_width=True)
            st.info("Sample output: NORMAL=0.30, PNEUMONIA=0.70")
        return

    # Load model
    try:
        model, idx_to_class, class_to_idx, img_size, model_name = load_model(Path(args.ckpt))
        pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    st.sidebar.write(f"**Model**: {model_name}")
    st.sidebar.write(f"**Image size**: {img_size}px")
    st.sidebar.write(f"**Classes**: {list(idx_to_class.values())}")
    
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
    threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05,
                                   help="Threshold for PNEUMONIA classification")
    
    # File uploader
    uploaded = st.file_uploader("📤 Upload chest X-ray image", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded:
        # Load and display original image
        img = Image.open(uploaded).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(img, use_container_width=True)
        
        # Prepare image for model
        tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = tf(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
        
        # Get prediction
        pred_idx = logits.argmax(dim=1).item()
        pred_class = idx_to_class[pred_idx]
        pred_conf = probs[pred_idx].item()
        
        # Threshold-based prediction for PNEUMONIA
        pneumonia_prob = probs[pneumonia_idx].item()
        threshold_pred = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
        
        # Display results
        with col2:
            st.subheader("📊 Prediction Results")
            
            # Show probabilities
            for idx in sorted(idx_to_class.keys()):
                class_name = idx_to_class[idx]
                prob = probs[idx].item()
                st.metric(label=class_name, value=f"{prob:.1%}")
            
            st.markdown("---")
            
            # Default prediction
            if pred_class == "PNEUMONIA":
                st.error(f"⚠️ **Predicted**: {pred_class} (confidence: {pred_conf:.1%})")
            else:
                st.success(f"✅ **Predicted**: {pred_class} (confidence: {pred_conf:.1%})")
            
            # Threshold-based prediction
            st.info(f"🎚️ **Threshold-based** (t={threshold:.2f}): {threshold_pred}")
            
            # Warning for borderline cases
            if 0.4 <= pneumonia_prob <= 0.6:
                st.warning("⚠️ Borderline case - consider additional clinical evaluation")
        
        # Grad-CAM visualization
        if show_gradcam:
            st.subheader("🔥 Grad-CAM Visualization")
            
            with st.spinner("Generating Grad-CAM..."):
                cam = generate_gradcam(model, x, pred_idx, model_name)
            
            if cam is not None:
                # Create overlay
                overlay_img = overlay_heatmap(img, cam, alpha=0.4)
                
                col_cam1, col_cam2 = st.columns(2)
                with col_cam1:
                    st.image(cam, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
                with col_cam2:
                    st.image(overlay_img, caption="Overlay", use_container_width=True)
                
                st.info("💡 Heatmap shows regions that influenced the model's decision (warmer colors = higher importance)")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.caption("🔬 Developed for educational and research purposes | CSE-4095 Deep Learning Project")


if __name__ == '__main__':
    main()
