import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import matplotlib.pyplot as plt

from models.nnunet import PlainUNet as nnUNet
from models.mk_unet import MKUNet
from models.configs.base_config import Config

config = Config()

st.set_page_config(
    page_title="甲状腺结节分割系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_configs = [
        ('nnUNet', nnUNet, config.MODEL_FEATURES['nnUNet']),
        ('MKUNet', MKUNet, config.MODEL_FEATURES['MKUNet']),
    ]

    for name, model_class, features in model_configs:
        model = model_class(in_channels=1, out_channels=2, features=features)
        model_path = f'results/{name}/best_model.pth'

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models[name] = model

    return models, device

def preprocess_image(image):
    img = np.array(image.convert('L'))
    img = Image.fromarray(img).resize((448, 384))
    img = np.array(img)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    mean = img.mean()
    std = img.std()
    img = (img - mean) / (std + 1e-8)
    return img

def segment_image(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return pred

def get_edge(mask):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    edges = np.array(edges)
    edges = (edges > 50).astype(np.uint8) * 255
    return edges

def create_comparison(image, mask):
    img_rgb = image.convert('RGB')
    img_rgb = img_rgb.resize((448, 384))
    img_rgb = np.array(img_rgb)

    mask_uint8 = (mask * 255).astype(np.uint8)

    overlay = img_rgb.copy()
    for i in range(3):
        if i == 0:
            overlay[:, :, i] = np.where(mask_uint8 > 0, 
                                       np.minimum(overlay[:, :, i] + 100, 255), 
                                       overlay[:, :, i])
        elif i == 2:
            overlay[:, :, i] = np.where(mask_uint8 > 0, 
                                       np.minimum(overlay[:, :, i] + 50, 255), 
                                       overlay[:, :, i])

    edges = get_edge(mask)
    for i in range(3):
        overlay[:, :, i] = np.where(edges > 0, 0, overlay[:, :, i])
    overlay[:, 1] = np.where(edges > 0, np.maximum(overlay[:, 1], 200), overlay[:, 1])

    return overlay

st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-box {
        text-align: center;
        padding: 15px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-zone {
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🩺 甲状腺结节超声图像分割系统</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📤 上传图像")
    uploaded_file = st.file_uploader(
        "点击或拖拽超声图像",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="支持 PNG / JPG 格式"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="原始图像", use_container_width=True)

    st.markdown("### 🔬 选择模型")
    model_choice = st.radio(
        "选择分割模型",
        ['nnUNet', 'MKUNet'],
        horizontal=True
    )

    run_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

    st.markdown("### 📊 模型信息")

    if model_choice == 'nnUNet':
        st.markdown("""
        <div class="model-card">
        <h4>nnUNet</h4>
        <p>基于自配置深度学习框架的分割模型</p>
        <p><strong>通道配置:</strong> [32, 64, 128, 256, 512, 512, 512]</p>
        <p><strong>训练轮数:</strong> 50 epochs</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-card">
        <h4>MKUNet</h4>
        <p>多核卷积U-Net，多尺度特征提取</p>
        <p><strong>通道配置:</strong> [32, 64, 128, 256, 512]</p>
        <p><strong>训练轮数:</strong> 50 epochs</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🔍 分析结果")

    if uploaded_file is None:
        st.info("👆 请先上传超声图像")
    elif not run_button:
        st.info("👆 点击「开始分析」按钮进行分割")
    else:
        with st.spinner('🤖 AI 分析中...'):
            models, device = load_models()

            if model_choice not in models:
                st.error(f"模型 {model_choice} 未找到，请先训练模型")
            else:
                img_tensor = preprocess_image(image)
                pred = segment_image(models[model_choice], img_tensor, device)
                pred_binary = (pred > 0).astype(np.uint8)

                result_img = create_comparison(image, pred_binary)

                st.image(result_img, caption=f"{model_choice} 分割结果", use_container_width=True)

                col_a, col_b, col_c, col_d = st.columns(4)

                preds_flat = pred_binary.flatten()
                targets_flat = np.zeros_like(preds_flat)

                TP = ((preds_flat == 1) & (targets_flat == 1)).sum()
                FP = ((preds_flat == 1) & (targets_flat == 0)).sum()
                FN = ((preds_flat == 0) & (targets_flat == 1)).sum()
                TN = ((preds_flat == 0) & (targets_flat == 0)).sum()

                dice = 2 * TP / (2 * TP + FP + FN + 1e-8) if (2 * TP + FP + FN) > 0 else 0
                iou = TP / (TP + FP + FN + 1e-8) if (TP + FP + FN) > 0 else 0

                with col_a:
                    st.metric("DICE", f"{dice:.4f}")
                with col_b:
                    st.metric("IOU", f"{iou:.4f}")
                with col_c:
                    area = pred_binary.sum()
                    st.metric("分割面积", f"{area}")
                with col_d:
                    perimeter = np.count_nonzero(get_edge(pred_binary))
                    st.metric("边缘点数", f"{perimeter}")

                st.success("✅ 分析完成！")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📋 系统说明

    本系统基于 nnU-Net V2 最优参数，对8种主流分割模型进行对比实验。
    当前演示包含 **nnUNet** 和 **MKUNet** 两个模型。

    **模型特点:**
    - **nnUNet**: 自适应配置框架，自动确定最佳预处理和训练参数
    - **MKUNet**: 多核卷积设计，更好的多尺度特征提取能力

    **技术参数:**
    - 输入尺寸: 384×448
    - 批次大小: 16
    - 初始学习率: 0.0001
    - 优化器: Adam
    - 损失函数: Dice + BCE
    """)
    st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🏥 基于深度学习的甲状腺结节超声图像分割系统</p>
    <p>使用 nnU-Net V2 最优参数 | 对比实验系统</p>
</div>
""", unsafe_allow_html=True)
