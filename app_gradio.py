import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import transforms

from models.nnunet import PlainUNet as nnUNet
from models.mk_unet import MKUNet
from models.configs.base_config import Config

config = Config()

print("正在加载模型...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_configs = [
    ('nnUNet', nnUNet, config.MODEL_FEATURES['nnUNet']),
    ('MKUNet', MKUNet, config.MODEL_FEATURES['MKUNet']),
]

models = {}
for name, model_class, features in model_configs:
    model = model_class(in_channels=1, out_channels=2, features=features)
    model_path = f'results/{name}/best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[name] = model
        print(f"✓ {name} 模型已加载")

def preprocess_image(image):
    img = np.array(image.convert('L'))
    img = cv2.resize(img, (448, 384))
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    mean = img.mean()
    std = img.std()
    img = (img - mean) / (std + 1e-8)
    return img

def segment_image(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return pred

def get_edge(mask):
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges

def create_overlay(image, mask):
    img_rgb = np.array(image.convert('RGB'))
    img_rgb_resized = cv2.resize(img_rgb, (448, 384))
    img_bgr = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR)

    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_colored = np.zeros_like(img_bgr)
    mask_colored[:, :, 0] = mask_uint8

    overlay = cv2.addWeighted(img_bgr, 0.7, mask_colored, 0.3, 0)

    edges = get_edge(mask)
    overlay[edges > 0] = [0, 255, 255]

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def process_single(image, model_name):
    if image is None:
        return None

    img_tensor = preprocess_image(image)
    pred = segment_image(models[model_name], img_tensor)
    pred_binary = (pred > 0).astype(np.uint8)
    overlay = create_overlay(image, pred_binary)

    return Image.fromarray(overlay)

def process_all(image):
    if image is None:
        return [None] * 2

    results = []
    for name in models.keys():
        img_tensor = preprocess_image(image)
        pred = segment_image(models[name], img_tensor)
        pred_binary = (pred > 0).astype(np.uint8)
        overlay = create_overlay(image, pred_binary)
        results.append(Image.fromarray(overlay))

    return results

with gr.Blocks(title="甲状腺结节分割系统") as demo:
    gr.Markdown("# 甲状腺结节超声图像分割系统")
    gr.Markdown("上传一张甲状腺超声图像，选择模型进行分割")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图像")
            model_select = gr.Dropdown(
                choices=list(models.keys()),
                value="nnUNet",
                label="选择模型"
            )
            single_btn = gr.Button("运行分割", variant="primary")

    with gr.Row():
        single_output = gr.Image(label="分割结果")

    single_btn.click(
        fn=process_single,
        inputs=[input_image, model_select],
        outputs=single_output
    )

    gr.Markdown("---")
    gr.Markdown("## 所有模型对比")

    all_btn = gr.Button("显示所有模型结果", variant="secondary")

    with gr.Row():
        outputs = [gr.Image(label=name) for name in models.keys()]

    all_btn.click(
        fn=process_all,
        inputs=input_image,
        outputs=outputs
    )

    gr.Markdown("---")
    gr.Markdown("### 支持的模型: " + ", ".join(models.keys()))

print("启动Gradio应用...")
demo.launch(server_name="0.0.0.0", server_port=7861)
