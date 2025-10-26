import argparse
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T

DISCLAIMER = "For research/education only. Not a medical device. Do not use for clinical diagnosis."


def load_model(ckpt_path: Path, num_classes: int = 3):
    from src.models.factory import build_model
    model, _ = build_model('resnet18', num_classes)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    idx_to_class = {v: k for k, v in ckpt['classes'].items()}
    return model, idx_to_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='runs/best.pt')
    parser.add_argument('--demo_mode', action='store_true')
    args, unknown = parser.parse_known_args()

    st.set_page_config(page_title="Pneumonia X-ray (Demo)", layout='wide')
    st.title("Pneumonia X-ray — Demo")
    st.info(DISCLAIMER)

    if args.demo_mode or not Path(args.ckpt).exists():
        st.warning("Demo mode: no model weights provided. Showing a static flow.")
        uploaded = st.file_uploader("Upload chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"]) 
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Input image", use_column_width=True)
            st.write("Sample output: Normal=0.30, Bacterial=0.40, Viral=0.30")
        return

    model, idx_to_class = load_model(Path(args.ckpt))

    uploaded = st.file_uploader("Upload chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"]) 
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Input image", use_column_width=True)
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
        for idx, p in enumerate(probs):
            st.write(f"{idx_to_class.get(idx, str(idx))}: {p:.3f}")


if __name__ == '__main__':
    main()
