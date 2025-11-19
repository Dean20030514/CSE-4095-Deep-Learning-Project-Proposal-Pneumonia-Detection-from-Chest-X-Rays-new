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
import pandas as pd

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
    
    # Determine target layer based on model architecture
    target_layer = None
    
    try:
        if 'resnet' in model_name.lower():
            # ResNet: use layer4 (last residual block)
            target_layer = 'layer4'
        elif 'efficientnet' in model_name.lower():
            # EfficientNet: use the last convolutional block
            # Try to find the correct layer name
            if hasattr(model, 'features'):
                # Count the number of blocks
                num_blocks = len(model.features)
                target_layer = f'features.{num_blocks - 1}'
            else:
                target_layer = 'features'
        elif 'densenet' in model_name.lower():
            # DenseNet: use features.denseblock4
            target_layer = 'features.denseblock4'
        else:
            # Default fallback
            target_layer = 'layer4'
        
        # Try to generate Grad-CAM
        gradcam = GradCAM(model, target_layer)
        model.eval()
        image_tensor.requires_grad = True
        # Pass image_tensor, not logits!
        cam = gradcam(image_tensor, target_class)
        return cam.cpu().numpy()
        
    except Exception as e:
        # If failed, try alternative methods
        alternative_layers = ['features.7', 'features.8', 'layer4', 'features']
        
        for alt_layer in alternative_layers:
            try:
                gradcam = GradCAM(model, alt_layer)
                model.eval()
                image_tensor.requires_grad = True
                # Pass image_tensor, not logits!
                cam = gradcam(image_tensor, target_class)
                return cam.cpu().numpy()
            except:
                continue
        
        # If all attempts failed
        st.warning(f"⚠️ Could not generate Grad-CAM for {model_name}. "
                  f"This model architecture may not be fully supported for visualization.")
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


def get_available_models():
    """Scan for available trained models"""
    runs_dir = Path('runs')
    if not runs_dir.exists():
        return []
    
    models = []
    for ckpt_path in runs_dir.glob('*/best_model.pt'):
        experiment_name = ckpt_path.parent.name
        models.append({
            'name': experiment_name,
            'path': str(ckpt_path),
            'display': f"{experiment_name}"
        })
    
    return sorted(models, key=lambda x: x['name'])


def get_model_performance(experiment_name):
    """Get test set performance for a model from analysis results"""
    perf_map = {
        'aug_aggressive': {'macro_recall': 97.39, 'pneumonia_recall': 97.18, 'accuracy': 97.30, 'pr_auc': 99.89},
        'model_densenet121': {'macro_recall': 98.45, 'pneumonia_recall': 98.11, 'accuracy': 98.29, 'pr_auc': 99.85},
        'lr_0.0001': {'macro_recall': 98.00, 'pneumonia_recall': 99.06, 'accuracy': 98.31, 'pr_auc': 99.90},
        'model_efficientnet_b0': {'macro_recall': 98.38, 'pneumonia_recall': 98.58, 'accuracy': 98.47, 'pr_auc': 99.87},
        'full_resnet18': {'macro_recall': 98.33, 'pneumonia_recall': 97.65, 'accuracy': 98.31, 'pr_auc': 99.86},
    }
    return perf_map.get(experiment_name, None)


def main():
    st.set_page_config(page_title="Pneumonia X-ray Detection", page_icon="🫁", layout='wide')
    
    # Initialize session state for uploader keys
    if 'single_uploader_key' not in st.session_state:
        st.session_state.single_uploader_key = 0
    if 'batch_uploader_key' not in st.session_state:
        st.session_state.batch_uploader_key = 0
    
    # Header
    st.title("🫁 Pneumonia Detection from Chest X-Rays")
    st.markdown(f"### {DISCLAIMER}")
    
    # Check for available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No trained models found in `runs/` directory")
        st.info("💡 **How to fix:**\n"
                "Run training first:\n"
                "```bash\n"
                "python src/train.py --config src/configs/model_efficientnet_b2.yaml\n"
                "```")
        return
    
    # Sidebar - Model Selection
    st.sidebar.header("🎯 Model Selection")
    
    # Model selector
    model_options = {m['display']: m for m in available_models}
    selected_display = st.sidebar.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        index=0,
        help="Select from available trained models"
    )
    
    selected_model = model_options[selected_display]
    ckpt_path = Path(selected_model['path'])
    
    # Load model
    try:
        with st.spinner(f"Loading model: {selected_model['name']}..."):
            model, idx_to_class, class_to_idx, img_size, model_name = load_model(ckpt_path)
            pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return
    
    # Display model information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Model Information")
    
    col_info1, col_info2 = st.sidebar.columns(2)
    with col_info1:
        st.markdown(f"**Architecture**")
        st.markdown(f"**Image Size**")
        st.markdown(f"**Experiment**")
    with col_info2:
        st.markdown(f"`{model_name}`")
        st.markdown(f"`{img_size}px`")
        st.markdown(f"`{selected_model['name']}`")
    
    # Show test set performance if available
    perf = get_model_performance(selected_model['name'])
    if perf:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Test Set Performance")
        col_p1, col_p2 = st.sidebar.columns(2)
        with col_p1:
            st.metric("Accuracy", f"{perf['accuracy']:.2f}%")
            st.metric("Macro Recall", f"{perf['macro_recall']:.2f}%")
        with col_p2:
            st.metric("Pneumonia Recall", f"{perf['pneumonia_recall']:.2f}%", delta="Primary KPI")
            st.metric("PR-AUC", f"{perf['pr_auc']:.2f}%")
    else:
        st.sidebar.info("ℹ️ Test performance not available for this model")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("❓ How do thresholds work?"):
        st.markdown("""
        **Threshold determines the decision boundary:**
        
        - Model outputs probability (0-100%)
        - If PNEUMONIA prob ≥ threshold → PNEUMONIA
        - If PNEUMONIA prob < threshold → NORMAL
        
        **When you'll see differences:**
        - ✅ Borderline cases (prob ~10-60%)
        - ❌ Extreme cases (prob >90% or <10%)
        
        **Example:**
        - Prob = 12% → Screening: PNEUMONIA, Confirmatory: NORMAL
        - Prob = 95% → All modes: PNEUMONIA
        - Prob = 3% → All modes: NORMAL
        
        💡 **Tip**: Upload borderline cases to see threshold effects!
        """)
    
    # Mode selection for threshold
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Operation Mode")
    
    mode = st.sidebar.selectbox(
        "Clinical Scenario",
        ["Balanced Mode", "Screening Mode (High Sensitivity)", "Confirmatory Mode (High Precision)", "Custom Threshold"],
        help="Choose preset thresholds optimized for different clinical scenarios"
    )
    
    # Set threshold based on mode (based on our threshold sweep results)
    mode_thresholds = {
        "Screening Mode (High Sensitivity)": 0.10,  # Max Recall: 99.06%
        "Balanced Mode": 0.15,  # Balanced F1: 98.60%
        "Confirmatory Mode (High Precision)": 0.525,  # Max Youden: 96.0% sensitivity, 98.8% specificity
        "Custom Threshold": 0.50
    }
    
    default_threshold = mode_thresholds[mode]
    
    if mode == "Custom Threshold":
        threshold = st.sidebar.slider(
            "Classification Threshold", 
            0.05, 0.95, default_threshold, 0.05,
            help="Probability threshold for PNEUMONIA classification"
        )
    else:
        threshold = default_threshold
        st.sidebar.info(f"📍 Threshold: **{threshold:.2f}**")
        
        # Add performance metrics for each mode (from test set results)
        if mode == "Screening Mode (High Sensitivity)":
            st.sidebar.success("✅ **Expected Performance:**\n"
                             "- Pneumonia Recall: ~99%\n"
                             "- Precision: ~97%\n"
                             "- Best for: Triage & Initial Screening")
        elif mode == "Balanced Mode":
            st.sidebar.success("✅ **Expected Performance:**\n"
                             "- Pneumonia Recall: ~99%\n"
                             "- Precision: ~98%\n"
                             "- Best for: General Clinical Use")
        elif mode == "Confirmatory Mode (High Precision)":
            st.sidebar.success("✅ **Expected Performance:**\n"
                             "- Pneumonia Recall: ~97%\n"
                             "- Precision: ~99.5%\n"
                             "- Best for: Confirmatory Testing")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Visualization Options")
    show_gradcam = st.sidebar.checkbox("Enable Grad-CAM", value=True, 
                                       help="Show visual explanation of model's decision")
    show_confidence = st.sidebar.checkbox("Show Confidence Analysis", value=True,
                                          help="Display probability distribution and confidence metrics")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Select Analysis Type",
        ["Single Image", "Batch Processing"],
        help="Choose between single image analysis or batch processing multiple images"
    )
    
    # File uploader
    if analysis_mode == "Single Image":
        col_single_upload, col_single_clear = st.columns([4, 1])
        with col_single_upload:
            uploaded = st.file_uploader("📤 Upload Chest X-ray Image", 
                                       type=["png", "jpg", "jpeg", "bmp"],
                                       key=f"single_uploader_{st.session_state.single_uploader_key}")
        with col_single_clear:
            if st.button("🔄 New Image", help="Clear current image and upload new one", type="secondary"):
                st.session_state.single_uploader_key += 1
                st.rerun()
    else:
        col_upload, col_clear = st.columns([4, 1])
        with col_upload:
            uploaded_files = st.file_uploader("📤 Upload Multiple Chest X-ray Images", 
                                              type=["png", "jpg", "jpeg", "bmp"], 
                                              accept_multiple_files=True,
                                              key=f"batch_uploader_{st.session_state.batch_uploader_key}")
        with col_clear:
            if st.button("🗑️ Clear All", help="Clear all uploaded images and reset", type="secondary"):
                st.session_state.batch_uploader_key += 1
                st.rerun()
        
        if uploaded_files:
            col_info, col_action = st.columns([3, 1])
            with col_info:
                st.info(f"📁 Uploaded {len(uploaded_files)} images")
            with col_action:
                if st.button("🔄 Process Again", help="Re-process the uploaded images"):
                    # Don't change key, just rerun to reprocess
                    st.rerun()
    
    # Single Image Analysis
    if analysis_mode == "Single Image" and uploaded:
        # Load and display original image
        img = Image.open(uploaded).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(img, use_container_width=True)
            st.caption(f"Filename: {uploaded.name}")
        
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
        normal_prob = probs[1 - pneumonia_idx].item()
        threshold_pred = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
        
        # Display results
        with col2:
            st.subheader("📊 Prediction Results")
            
            # Show probabilities with progress bars
            st.markdown("**Class Probabilities:**")
            for idx in sorted(idx_to_class.keys()):
                class_name = idx_to_class[idx]
                prob = probs[idx].item()
                col_name, col_prob, col_bar = st.columns([2, 1, 3])
                with col_name:
                    st.write(f"**{class_name}**")
                with col_prob:
                    st.metric("", f"{prob:.1%}")
                with col_bar:
                    st.progress(prob)
            
            st.markdown("---")
            
            # Threshold-based prediction (primary result)
            st.markdown(f"### 🎯 Clinical Decision (Threshold={threshold:.2f})")
            if threshold_pred == "PNEUMONIA":
                st.error(f"### ⚠️ PNEUMONIA DETECTED\n**Confidence:** {pneumonia_prob:.1%}")
                st.markdown("**🏥 Clinical Recommendation:**")
                if pneumonia_prob >= 0.90:
                    st.warning("- **High confidence positive**\n"
                             "- Recommend immediate clinical correlation\n"
                             "- Consider treatment initiation if clinically indicated\n"
                             "- Follow-up imaging if clinically appropriate")
                elif pneumonia_prob >= 0.75:
                    st.info("- **Moderate-high confidence positive**\n"
                           "- Recommend clinical evaluation\n"
                           "- Consider confirmatory testing or imaging\n"
                           "- Monitor patient closely")
                else:
                    st.info("- **Low-moderate confidence positive**\n"
                           "- Recommend additional evaluation\n"
                           "- Consider repeat imaging or alternative diagnostic methods\n"
                           "- Clinical correlation essential")
            else:
                st.success(f"### ✅ NO PNEUMONIA DETECTED\n**Confidence:** {normal_prob:.1%}")
                st.markdown("**🏥 Clinical Recommendation:**")
                if normal_prob >= 0.90:
                    st.info("- **High confidence negative**\n"
                           "- Pneumonia unlikely based on imaging\n"
                           "- Continue standard clinical evaluation\n"
                           "- Consider alternative diagnoses if symptoms persist")
                elif normal_prob >= 0.75:
                    st.info("- **Moderate-high confidence negative**\n"
                           "- Pneumonia less likely\n"
                           "- Clinical correlation recommended\n"
                           "- Follow-up if symptoms worsen")
                else:
                    st.warning("- **Low confidence result**\n"
                             "- Inconclusive findings\n"
                             "- **Recommend repeat imaging or additional testing**\n"
                             "- Do not rule out pneumonia based on this result alone")
            
            # Warning for borderline cases
            if 0.35 <= pneumonia_prob <= 0.65:
                st.warning("⚚ **BORDERLINE CASE - UNCERTAIN RESULT**\n\n"
                         "This case falls in the uncertain range. The model cannot confidently classify this image.\n\n"
                         "**Strongly recommend:**\n"
                         "- Clinical correlation with patient symptoms and history\n"
                         "- Additional diagnostic tests (labs, repeat imaging, CT scan)\n"
                         "- Expert radiologist review\n"
                         "- Do NOT rely solely on this automated result")
            
            # Show threshold impact comparison (for educational purposes)
            st.markdown("---")
            with st.expander("🎚️ **How different thresholds affect this case**", expanded=False):
                st.markdown(f"**Current Pneumonia Probability: {pneumonia_prob:.2%}**")
                st.markdown("")
                
                # Compare all modes
                threshold_comparison = {
                    "Screening Mode (t=0.10)": 0.10,
                    "Balanced Mode (t=0.15)": 0.15,
                    "Confirmatory Mode (t=0.525)": 0.525,
                }
                
                col_t1, col_t2, col_t3 = st.columns(3)
                
                for i, (mode_name, t_val) in enumerate(threshold_comparison.items()):
                    decision = "PNEUMONIA" if pneumonia_prob >= t_val else "NORMAL"
                    
                    with [col_t1, col_t2, col_t3][i]:
                        if decision == "PNEUMONIA":
                            st.error(f"**{mode_name}**")
                            st.write(f"Result: ⚠️ **PNEUMONIA**")
                        else:
                            st.success(f"**{mode_name}**")
                            st.write(f"Result: ✅ **NORMAL**")
                        
                        # Show if this is the current selection
                        if abs(threshold - t_val) < 0.001:
                            st.info("👈 Current selection")
                
                st.markdown("---")
                st.caption("💡 **Interpretation**: Different thresholds can produce different results for borderline cases. "
                         "Choose the mode based on your clinical scenario.")
        
        # Confidence Analysis
        if show_confidence:
            st.markdown("---")
            st.subheader("📈 Confidence Analysis")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Probability distribution chart with threshold markers
                fig, ax = plt.subplots(figsize=(7, 4))
                classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                probabilities = [probs[i].item() for i in sorted(idx_to_class.keys())]
                colors = ['#2ecc71' if c == 'NORMAL' else '#e74c3c' for c in classes]
                
                bars = ax.barh(classes, probabilities, color=colors, alpha=0.7, height=0.4)
                
                # Mark all three mode thresholds
                ax.axvline(x=0.10, color='#e67e22', linestyle=':', linewidth=1.5, alpha=0.5, label='Screening (0.10)')
                ax.axvline(x=0.15, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.5, label='Balanced (0.15)')
                ax.axvline(x=0.525, color='#9b59b6', linestyle=':', linewidth=1.5, alpha=0.5, label='Confirmatory (0.525)')
                
                # Highlight current threshold
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2.5, label=f'CURRENT ({threshold:.2f})')
                
                ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
                ax.set_xlim([0, 1])
                ax.set_title('Prediction Confidence vs Thresholds', fontsize=13, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for bar, prob in zip(bars, probabilities):
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1%}', ha='left', va='center', fontsize=10, fontweight='bold')
                
                # Add decision region shading for current threshold
                ax.axvspan(threshold, 1.0, alpha=0.1, color='red', label='PNEUMONIA Region')
                ax.axvspan(0, threshold, alpha=0.1, color='green', label='NORMAL Region')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col_chart2:
                # Confidence metrics
                confidence_level = "High" if max(probabilities) >= 0.85 else "Medium" if max(probabilities) >= 0.65 else "Low"
                uncertainty = 1 - max(probabilities)
                
                st.markdown("**Decision Metrics:**")
                st.metric("Confidence Level", confidence_level)
                st.metric("Uncertainty", f"{uncertainty:.1%}")
                st.metric("Prediction Margin", f"{abs(pneumonia_prob - normal_prob):.1%}")
                
                st.markdown("---")
                st.markdown("**Interpretation Guide:**")
                st.markdown("- **High confidence** (>85%): Model very certain")
                st.markdown("- **Medium confidence** (65-85%): Reasonably confident")
                st.markdown("- **Low confidence** (<65%): Uncertain, needs review")
        
        # Grad-CAM visualization
        if show_gradcam:
            st.markdown("---")
            st.subheader("🔥 Grad-CAM Explainability")
            st.caption("Visual explanation showing which image regions influenced the model's prediction")
            
            with st.spinner("Generating Grad-CAM visualization..."):
                cam = generate_gradcam(model, x, pneumonia_idx, model_name)
            
            if cam is not None:
                # Create overlay
                overlay_img = overlay_heatmap(img, cam, alpha=0.4)
                
                # Three-panel display
                col_cam1, col_cam2, col_cam3 = st.columns(3)
                
                with col_cam1:
                    st.markdown("**📷 Original X-ray**")
                    st.image(img, use_container_width=True)
                
                with col_cam2:
                    st.markdown("**🌡️ Attention Heatmap**")
                    st.image(cam, use_container_width=True, clamp=True)
                
                with col_cam3:
                    st.markdown(f"**🔍 Overlay ({threshold_pred})**")
                    st.image(overlay_img, use_container_width=True)
                
                st.info("💡 **How to interpret Grad-CAM:**\n"
                       "- **Red/Yellow regions**: Areas the model focuses on most strongly\n"
                       "- **Blue/Dark regions**: Areas with minimal influence on the decision\n"
                       "- For pneumonia cases, look for activation in lung fields\n"
                       "- Artifacts (tubes, wires, labels) should ideally have low activation")
    
    # Batch Processing Mode
    elif analysis_mode == "Batch Processing" and 'uploaded_files' in locals() and uploaded_files:
        st.markdown("---")
        st.subheader(f"📋 Batch Analysis Results ({len(uploaded_files)} images)")
        
        # Process all images
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
            
            img = Image.open(file).convert('RGB')
            x = tf(img).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0)
            
            pneumonia_prob = probs[pneumonia_idx].item()
            normal_prob = probs[1 - pneumonia_idx].item()
            threshold_pred = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
            
            results.append({
                'Filename': file.name,
                'Prediction': threshold_pred,
                'PNEUMONIA Probability': f"{pneumonia_prob:.2%}",
                'NORMAL Probability': f"{normal_prob:.2%}",
                'Confidence': f"{max(pneumonia_prob, normal_prob):.2%}"
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Batch processing complete!")
        
        # Display results table
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Batch Summary")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        pneumonia_count = sum(1 for r in results if r['Prediction'] == 'PNEUMONIA')
        normal_count = len(results) - pneumonia_count
        
        with col_s1:
            st.metric("Total Images", len(results))
        with col_s2:
            st.metric("PNEUMONIA Cases", pneumonia_count, 
                     delta=f"{pneumonia_count/len(results)*100:.1f}%")
        with col_s3:
            st.metric("NORMAL Cases", normal_count,
                     delta=f"{normal_count/len(results)*100:.1f}%")
        with col_s4:
            avg_conf = sum(float(r['Confidence'].strip('%'))/100 for r in results) / len(results)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        # Distribution chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart
        ax1.pie([pneumonia_count, normal_count], labels=['PNEUMONIA', 'NORMAL'],
               colors=['#e74c3c', '#2ecc71'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Prediction Distribution', fontweight='bold')
        
        # Confidence histogram
        confidences = [float(r['PNEUMONIA Probability'].strip('%'))/100 for r in results]
        ax2.hist(confidences, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        ax2.set_xlabel('PNEUMONIA Probability', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Confidence Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Action buttons
        st.markdown("---")
        col_action1, col_action2, col_action3 = st.columns(3)
        
        with col_action1:
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"pneumonia_batch_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_action2:
            # Clear and start over
            if st.button("🗑️ Clear All & Start Over", use_container_width=True, type="secondary"):
                st.session_state.batch_uploader_key += 1
                st.rerun()
        
        with col_action3:
            # Show processing info
            st.metric("Processing Status", "✅ Complete", 
                     delta=f"{len(uploaded_files)} images analyzed")
    
    else:
        st.info("👆 Please upload chest X-ray image(s) to begin analysis")
        
        # Show example
        with st.expander("ℹ️ How to use this tool"):
            st.markdown("""
            **Single Image Mode:**
            1. Select "Single Image" in the sidebar
            2. Upload a chest X-ray image (PNG/JPG/JPEG/BMP)
            3. Choose an operation mode (Screening/Balanced/Confirmatory)
            4. Review the prediction, confidence analysis, and Grad-CAM visualization
            5. Follow clinical recommendations provided
            
            **Batch Processing Mode:**
            1. Select "Batch Processing" in the sidebar
            2. Upload multiple chest X-ray images
            3. Review batch results in the table and summary charts
            4. Download results as CSV for further analysis
            
            **Operation Modes:**
            - **Screening Mode (High Sensitivity)**: Optimized to catch pneumonia cases (minimizes false negatives)
            - **Balanced Mode**: Good balance between sensitivity and specificity
            - **Confirmatory Mode (High Precision)**: Reduces false positives (for confirmatory testing)
            - **Custom Threshold**: Manually adjust the decision threshold
            
            **Important Notes:**
            - This is a research/educational tool, NOT a medical device
            - Do NOT use for actual clinical diagnosis
            - Always consult qualified healthcare professionals
            - Results should be validated by radiologists
            """)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.caption("🔬 **Pneumonia Detection System**")
        st.caption("CSE-4095 Deep Learning Project")
    with col_f2:
        st.caption(f"**Current Model:** {selected_model['name']}")
        st.caption(f"**Architecture:** {model_name} @ {img_size}px")
    with col_f3:
        st.caption("⚠️ **Research & Educational Use Only**")
        st.caption("Not FDA approved for clinical diagnosis")


if __name__ == '__main__':
    main()
