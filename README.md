This project implements Neural Style Transfer (NST) using PyTorch and VGG19, leveraging the MPS backend on MacBooks with Apple Silicon (M1/M2/M3) for GPU acceleration to blend the artistic style of one image with the content of another. The model extracts deep features from both images and optimizes a new image that combines the structure of the content image with the visual patterns of the style image.
Additionally, post-processing techniques such as gamma correction and overlay blending are applied to fine-tune the final output for a more polished look.

Features:
1. Uses PyTorch’s MPS backend for Mac GPU acceleration
2. Implements content and style loss for style transfer
3. Supports adjustable blending and gamma correction
4. Optimized for Apple Silicon (M1/M2/M3)
   
Dependencies:
1. Python 3.8+
2. PyTorch with MPS support (torch and torchvision)
3. NumPy, PIL, Matplotlib
