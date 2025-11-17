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
            st.image(img, caption="Input image", width='stretch')
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
    
    # Mode selection for threshold
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Operation Mode")
    
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Balanced Mode", "Screening Mode (High Sensitivity)", "High Precision Mode", "Custom"],
        help="Choose preset thresholds optimized for different scenarios"
    )
    
    # Set threshold based on mode
    mode_thresholds = {
        "Screening Mode (High Sensitivity)": 0.15,  # High recall for pneumonia
        "Balanced Mode": 0.50,  # Balance precision and recall
        "High Precision Mode": 0.75,  # Minimize false positives
        "Custom": 0.50
    }
    
    default_threshold = mode_thresholds[mode]
    
    if mode == "Custom":
        threshold = st.sidebar.slider(
            "Classification Threshold", 
            0.05, 0.95, default_threshold, 0.05,
            help="Threshold for PNEUMONIA classification"
        )
    else:
        threshold = default_threshold
        st.sidebar.info(f"Using threshold: {threshold:.2f}")
        if mode == "Screening Mode (High Sensitivity)":
            st.sidebar.caption("🔍 Optimized for catching pneumonia cases (minimizes false negatives)")
        elif mode == "High Precision Mode":
            st.sidebar.caption("🎯 Optimized for reducing false alarms")
    
    st.sidebar.markdown("---")
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
    
    # File uploader
    uploaded = st.file_uploader("📤 Upload chest X-ray image", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded:
        # Load and display original image
        img = Image.open(uploaded).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(img, width='stretch')
        
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
        pred_conf = float(probs[pred_idx].item())
        
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
            
            # Default prediction with confidence
            st.markdown("### 🎯 Model Prediction")
            if pred_class == "PNEUMONIA":
                st.error(f"⚠️ **{pred_class}** (confidence: {pred_conf:.1%})")
            else:
                st.success(f"✅ **{pred_class}** (confidence: {pred_conf:.1%})")
            
            # Threshold-based prediction
            st.markdown(f"### 🎚️ Mode-Based Decision (t={threshold:.2f})")
            if threshold_pred == "PNEUMONIA":
                st.error(f"⚠️ **{threshold_pred}** (PNEUMONIA probability: {pneumonia_prob:.1%})")
            else:
                st.success(f"✅ **{threshold_pred}** (PNEUMONIA probability: {pneumonia_prob:.1%})")
            
            # Warning for borderline cases
            if 0.35 <= pneumonia_prob <= 0.65:
                st.warning("⚠️ **Borderline case** - Consider additional clinical evaluation or repeat imaging")
        
        # Grad-CAM visualization
        if show_gradcam:
            st.markdown("---")
            st.subheader("🔥 Grad-CAM Visualization")
            st.caption("Visual explanation showing regions that influenced the model's decision")
            
            with st.spinner("Generating Grad-CAM..."):
                cam = generate_gradcam(model, x, pred_idx, model_name)
            
            if cam is not None:
                # Create overlay
                overlay_img = overlay_heatmap(img, cam, alpha=0.4)
                
                # Three-panel display
                col_cam1, col_cam2, col_cam3 = st.columns(3)
                
                with col_cam1:
                    st.markdown("**📷 Original Image**")
                    st.image(img, width='stretch')
                
                with col_cam2:
                    st.markdown("**🌡️ Attention Heatmap**")
                    st.image(cam, caption="", width='stretch', clamp=True)
                
                with col_cam3:
                    st.markdown(f"**🔍 Overlay (Prediction: {pred_class})**")
                    st.image(overlay_img, width='stretch')
                
                st.info("💡 **Interpretation**: Warmer colors (red/yellow) indicate regions with higher influence on the prediction. "
                       "The model focuses on these areas when making its decision.")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.caption("🔬 Developed for educational and research purposes | CSE-4095 Deep Learning Project")


if __name__ == '__main__':
    main()
