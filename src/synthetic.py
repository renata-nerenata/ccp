import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch
import cv2

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

from model.cluster_counter import ColorDistributionExtractor
from model.cluster_counter import count_model
from model.segmentation import seg_model

def generate_clusters(centers, covariances, sizes):
    clusters = []
    for center, cov, size in zip(centers, covariances, sizes):
        cluster = np.random.multivariate_normal(center, cov, size)
        clusters.append(cluster)
    return np.vstack(clusters)

def points_to_image(points, size=(256, 256)):
    img = np.zeros(size, dtype=np.uint8)
    norm_points = (points - points.min(0)) / (points.max(0) - points.min(0) + 1e-8)
    coords = (norm_points * (np.array(size) - 1)).astype(int)
    for x, y in coords:
        img[y, x] = 255
    return np.stack([img] * 3, axis=-1) / 255.0


fixed_centers = [[2, 2], [4, 4]]
fixed_sizes = [2000, 2000]
base_cov = np.array([[1, 0], [0, 1]])
scale_factors = [0.2, 0.5, 1.0, 1.5, 2.0]

color_map = {
    1: "#e0f3f8",
    2: "#a8dadc",
    3: "#457b9d"
}

fig, axs = plt.subplots(len(scale_factors), len(scale_factors), figsize=(15, 15), facecolor='none')

for i, scale1 in enumerate(scale_factors):
    for j, scale2 in enumerate(scale_factors):
        ax = axs[i, j]

        cov1 = base_cov * scale1
        cov2 = base_cov * scale2
        data = generate_clusters(fixed_centers, [cov1, cov2], fixed_sizes)
        img = points_to_image(data)

        color_extractor = ColorDistributionExtractor()
        color_feat = color_extractor.extract([img])
        count_pred = count_model.predict([np.expand_dims(img, 0), color_feat])
        count_scalar = np.argmax(count_pred, axis=1)[0] + 1

        mask_pred = seg_model.predict([
            np.expand_dims(img, 0), color_feat, np.array([[count_scalar]])
        ])[0, :, :, 0]

        ax.imshow(mask_pred, cmap='gray')
        ax.set_title(f"Pred n={count_scalar}\nS1={scale1}, S2={scale2}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(color_map[min(count_scalar, 3)])

plt.tight_layout()
legend_elements = [Patch(facecolor=color_map[n], label=f'n={n}') for n in [1, 2, 3]]
fig.legend(handles=legend_elements, loc='center right', title='Predicted Clusters')
plt.show()

