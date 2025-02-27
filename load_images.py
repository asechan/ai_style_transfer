import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Enable Metal GPU acceleration for M3 MacBook Air
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_img(path_to_img):
    max_dim = 1024  # Increased for better quality
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def blend_images(original, stylized, alpha=0.6):
    return original * (1 - alpha) + stylized * alpha

content_path = '_DSC4597.jpg'
style_path = 'IMG-20250131-WA0059.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

# Load a different model checkpoint
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Resize stylized image to match content image dimensions
stylized_image = tf.image.resize(stylized_image, tf.shape(content_image)[1:3])

# Blend the images by removing the batch dimension from both
blended_image = blend_images(content_image[0], stylized_image[0], alpha=0.6)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(tensor_to_image(content_image[0]))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tensor_to_image(stylized_image))
plt.title('Fully Stylized')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(tensor_to_image(blended_image))
plt.title('Blended Result')
plt.axis('off')

plt.show()

output_image = tensor_to_image(blended_image)
output_image.save('stylized_output.jpg')
