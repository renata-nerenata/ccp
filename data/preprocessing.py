import os
import zipfile
import numpy as np
from PIL import Image
import glob

IMG_WIDTH = 256
IMG_HEIGHT = 256


class DatasetPreparer:
    def __init__(self, zip_path: str, extract_dir: str):
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.input_dir = None
        self.mask_dir = None

    def unzip_and_prepare(self):
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)
        self.input_dir = os.path.join(self.extract_dir, 'data 2/input_images')
        self.mask_dir = os.path.join(self.extract_dir, 'data 2/output_images')

    def get_dirs(self):
        return self.input_dir, self.mask_dir


def augment_images(directory: str, prefix: str = "", angles=[90, 180, 270]):
    image_paths = glob.glob(os.path.join(directory, '*'))
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        img = Image.open(image_path)
        for angle in angles:
            rotated = img.rotate(angle, expand=False)
            save_path = os.path.join(directory, f"{prefix}{base_name}_rot{angle}{ext}")
            rotated.save(save_path)


def load_images_and_masks(image_dir, mask_dir):
    images, masks, filenames = [], [], []
    image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))])
    mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg'))])
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_filenames}
    for image_filename in image_filenames:
        base = os.path.splitext(image_filename)[0]
        if base in mask_dict:
            mask_filename = mask_dict[base]
            img_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, mask_filename)
            img = np.array(Image.open(img_path).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))) / 255.0
            mask = np.array(Image.open(mask_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            images.append(img)
            masks.append(mask)
            filenames.append((image_filename, mask_filename))
    return np.array(images), np.array(masks), filenames
