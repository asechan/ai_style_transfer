import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np

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

def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert("RGB")
    aspect_ratio = image.width / image.height
    if (aspect_ratio > 1):
        new_width, new_height = max_size, int(max_size / aspect_ratio)
    else:
        new_width, new_height = int(max_size * aspect_ratio), max_size
    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.unsqueeze(0))
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
    # tensor: feature maps from a particular layer
    gram = gram_matrix(tensor)
    # Use Frobenius norm as a proxy for texture / grain strength
    return torch.norm(gram, p='fro')

def load_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
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
                   epochs=20, style_weight=1, content_weight=1e4, lr=0.002):
    target_image = content_image.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([target_image], lr=lr)  
    tv_weight = 0.0001  
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
        
        grain_values = {}
        for layer in style_layers:
            style_grain = compute_grain_measure(get_features(style_image, vgg, [layer])[layer])
            target_grain = compute_grain_measure(get_features(target_image, vgg, [layer])[layer])
            grain_values[layer] = {'style': style_grain.item(), 'target': target_grain.item()}
        print(f"Grain values: {grain_values}")
        
    return target_image

def main():
    content_image = load_image("IMG-20240223-WA0015.jpg", max_size=1024)
    style_image = load_image("IMG-20250131-WA0059.jpg", max_size=1024)
    vgg = load_vgg()
    content_layers = ['21',]  
    style_layers = ['0', '5', '10', '19', '28']  
    result = style_transfer(content_image, style_image, vgg, content_layers, style_layers, 
                           epochs=20, style_weight=1, content_weight=1e4)
    result_image = tensor_to_image(result)
    result_image = adjust_gamma(result_image, gamma=0.8)
    original_image = Image.open("IMG-20240223-WA0015.jpg").convert("RGB")
    blended_result = overlay_blend(original_image, result_image, alpha=0.8)
    plt.imshow(blended_result)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
