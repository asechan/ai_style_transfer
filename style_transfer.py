import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def adjust_gamma(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = [int(((i / 255.0) ** invGamma) * 255) for i in range(256)]
    if image.mode in ('RGB', 'RGBA', 'CMYK'):
        table = table * len(image.getbands())
    return image.point(table)

def overlay_blend(original, styled, alpha=0.6):
    original = original.resize(styled.size, Image.LANCZOS)  
    orig = np.array(original).astype(np.float32) / 255.0
    styl = np.array(styled).astype(np.float32) / 255.0   
    blended = np.where(orig <= 0.5, 2 * orig * styl, 1 - 2 * (1 - orig) * (1 - styl))   
    result = (1 - alpha) * orig + alpha * blended
    result = np.clip(result * 255, 0, 255).astype(np.uint8)    
    return Image.fromarray(result)

def load_image(image_path, max_size=None):
    image = Image.open(image_path).convert("RGB")
    if max_size is not None:
        scale = max(image.size) / max_size
        if scale > 1:
            new_size = (int(image.size[0] / scale), int(image.size[1] / scale))
            image = image.resize(new_size, Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.unsqueeze(0).to(device))
    ])
    return transform(image)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu() 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

def compute_grain_measure(tensor):
    gram = gram_matrix(tensor)
    return torch.norm(gram, p='fro')

def load_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def compute_content_loss(target, content):
    return torch.nn.functional.mse_loss(target, content) / target.numel()

def compute_style_loss(gram_target, gram_style):
    return torch.nn.functional.mse_loss(gram_target, gram_style)

def tv_loss(img):
    loss = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return loss

def style_transfer(content_image, style_image, vgg, content_layers, style_layers, 
                   epochs=20, style_weight=1, content_weight=1e3, lr=0.002):
    target_image = content_image.clone().detach().requires_grad_(True).to(device)
    optimizer = optim.Adam([target_image], lr=lr)  
    tv_weight = 0.01  
    for epoch in range(epochs):
        optimizer.zero_grad()
        target_features = get_features(target_image, vgg, content_layers + style_layers)
        content_features = get_features(content_image, vgg, content_layers)
        style_features = get_features(style_image, vgg, style_layers)
        content_loss = sum(compute_content_loss(target_features[l], content_features[l]) for l in content_layers)    
        style_loss = 0
        for layer in style_layers:
            gram_t = gram_matrix(target_features[layer])
            gram_s = gram_matrix(style_features[layer])
            style_loss += compute_style_loss(gram_t, gram_s)
        total_loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss(target_image)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            target_image.clamp_(0,1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}")
    return target_image

def main():
    content_image_path = "path to content image"
    style_image_path = "path to style image"
    content_image = load_image(content_image_path,max_size=2048)
    style_image = load_image(style_image_path,max_size=2048)
    vgg = load_vgg()
    content_layers = ['21',]
    style_layers = ['0', '1', '3', '5', '10', '19', '28']
    result = style_transfer(content_image, style_image, vgg, content_layers, style_layers, epochs=20, style_weight=1, content_weight=1e3)
    result_image = tensor_to_image(result)
    result_image = adjust_gamma(result_image, gamma=0.8)
    
    original_image = Image.open(content_image_path).convert("RGB")
    blended_result = overlay_blend(original_image, result_image, alpha=0.6)
    
    base, ext = os.path.splitext(content_image_path)
    new_filename = f"{base}_stylized{ext}"
    blended_result.save(new_filename)
    plt.imshow(blended_result)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
