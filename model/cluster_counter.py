import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

class ColorDistributionExtractor:
    def __init__(self, sat_thresh=50, val_thresh=50):
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

    def _to_hsv(self, img_rgb):
        img_bgr = (img_rgb * 255).astype(np.uint8)[..., ::-1]
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    def _extract_ratios(self, hsv):
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        red_mask = (((hue < 10) | (hue > 170)) & (sat > self.sat_thresh) & (val > self.val_thresh))
        yellow_mask = ((hue >= 20) & (hue <= 35) & (sat > self.sat_thresh) & (val > self.val_thresh))
        blue_mask = ((hue >= 100) & (hue <= 130) & (sat > self.sat_thresh) & (val > self.val_thresh))

        total_pixels = hsv.shape[0] * hsv.shape[1]
        r_ratio = np.sum(red_mask) / total_pixels
        y_ratio = np.sum(yellow_mask) / total_pixels
        b_ratio = np.sum(blue_mask) / total_pixels

        return [r_ratio, y_ratio, b_ratio]

    def extract(self, images):
        features = []
        for img in images:
            hsv = self._to_hsv(img)
            ratios = self._extract_ratios(hsv)
            features.append(ratios)
        return np.array(features, dtype=np.float32)


import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

class ClusterCountTrainer:
    def __init__(self, images, masks, img_shape=(256, 256, 3), sat_thresh=50, val_thresh=50):
        self.images = images
        self.masks = masks
        self.img_shape = img_shape
        self.color_extractor = ColorDistributionExtractor(sat_thresh, val_thresh)

        self.color_features = self.color_extractor.extract(images)
        self.labels = self._compute_cluster_classes(masks)

        self.X_train_img, self.X_val_img, \
        self.X_train_col, self.X_val_col, \
        self.y_train, self.y_val = train_test_split(
            self.images, self.color_features, self.labels,
            test_size=0.2, random_state=42
        )

        self.model = self._build_model()

    def _compute_cluster_classes(self, masks):
        raw_counts = np.array([self._get_cluster_count(mask) for mask in masks], dtype=np.int32)
        clipped = np.clip(raw_counts, 1, 6)
        classes = clipped - 1
        return to_categorical(classes, num_classes=6)

    def _get_cluster_count(self, image_input):
        if image_input.ndim == 3:
            if image_input.shape[2] == 3:
                image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif image_input.shape[2] == 1:
                image = np.squeeze(image_input, axis=-1)
            else:
                raise ValueError("Unexpected number of channels.")
        else:
            image = image_input
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        inverted = cv2.bitwise_not(image)
        _, thresh = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    def _build_model(self):
        image_input = layers.Input(shape=self.img_shape)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)

        color_input = layers.Input(shape=(3,))
        combined = layers.Concatenate()([x, color_input])

        x = layers.Dense(64, activation='relu')(combined)
        output = layers.Dense(6, activation='softmax')(x)

        model = models.Model(inputs=[image_input, color_input], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=20, batch_size=8):
        self.model.fit(
            [self.X_train_img, self.X_train_col], self.y_train,
            validation_data=([self.X_val_img, self.X_val_col], self.y_val),
            batch_size=batch_size,
            epochs=epochs
        )

    def predict(self, images=None):
        if images is None:
            images = self.images
            color_feats = self.color_features
        else:
            color_feats = self.color_extractor.extract(images)
        probs = self.model.predict([images, color_feats])
        return probs


if __name__ == "__main__":
    trainer = ClusterCountTrainer(images=X, masks=Y, img_shape=(256, 256, 3))
    trainer.train(epochs=20)
    probabilities = trainer.predict()
    predicted_classes = np.argmax(probabilities, axis=1)
    predicted_counts = predicted_classes + 1

