import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import math
from matplotlib.patches import Ellipse
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error
import cv2
import io
from PIL import Image

def generate_image(k, a, b, r, xlim1 = -1.5, xlim2 = 1.5, ylim1 = -1.5, ylim2 = 1.5):
    circles = [Circle((x, y), radius, color='black', fill=False, linewidth=0.1)
               for x, y, radius in zip(a, b, r)]
    
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="w")
    ax.add_collection(PatchCollection(circles, match_original=True))
    ax.set_xlim(xlim1, xlim2)
    ax.set_ylim(ylim1, ylim2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close(fig)
    return img

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (256, 256))
    return img_gray

def calculate_metrics(ref, img):
    metrics = {}
    metrics['SSIM'] = ssim(ref, img) if len(ref.shape) == 2 else ssim(ref, img, channel_axis=2)
    metrics['MSE'] = mean_squared_error(ref, img)
    metrics['MAE'] = mean_absolute_error(ref.flatten(), img.flatten())
    metrics['PSNR'] = peak_signal_noise_ratio(ref, img)
    metrics['NCC'] = np.corrcoef(ref.flatten(), img.flatten())[0, 1]
    return metrics

def plot_comparison(images, results):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Original', 'Closer', 'Different', 'Completely Different']

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis('off')
        title = titles[i]
        if i > 0:
            metrics = results[i]
            subtitle = (f"SSIM: {metrics['SSIM']:.3f}\n"
                        f"MSE: {metrics['MSE']:.2f}\n"
                        f"MAE: {metrics['MAE']:.2f}\n"
                        f"PSNR: {metrics['PSNR']:.2f}\n"
                        f"NCC: {metrics['NCC']:.2f}")
            ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        else:
            ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.show()

def generate_pixel_img(rgb_func, width, height, c=[1,2,3]):
    m = np.linspace(1, width, width)
    n = np.linspace(1, height, height)
    m_grid, n_grid = np.meshgrid(m, n)

    r = rgb_func(m_grid, n_grid, c[0])
    g = rgb_func(m_grid, n_grid, c[1])
    b = rgb_func(m_grid, n_grid, c[2])

    rgb_image = np.stack([r, g, b], axis=-1).astype(np.uint8)
    
    return rgb_image