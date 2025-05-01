import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from model.cluster_counter import ColorDistributionExtractor

class UNetSegmentationTrainer:
    def __init__(self, images, masks, predicted_counts, sat_thresh=50, val_thresh=50):
        self.images = images
        self.masks = masks
        self.predicted_counts = predicted_counts.reshape(-1, 1)
        self.color_extractor = ColorDistributionExtractor(sat_thresh, val_thresh)

        self.color_features = self.color_extractor.extract(images)
        self.X_train_img, self.X_val_img, \
        self.X_train_col, self.X_val_col, \
        self.X_train_count, self.X_val_count, \
        self.Y_train, self.Y_val = train_test_split(
            self.images,
            self.color_features,
            self.predicted_counts,
            self.masks,
            test_size=0.2, random_state=42
        )

        self.model = self._build_unet(input_shape=images.shape[1:], color_shape=(3,), count_shape=(1,))

    def _build_unet(self, input_shape, color_shape, count_shape):
        image_input = layers.Input(shape=input_shape)
        color_input = layers.Input(shape=color_shape)
        count_input = layers.Input(shape=count_shape)

        x = layers.Conv2D(32, 3, activation='relu', padding='same')(image_input)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        p1 = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        p2 = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        p3 = layers.MaxPooling2D()(x)

        bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
        bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')(bottleneck)

        # Flatten context (color + count), tile spatially, concatenate to bottleneck
        context = layers.Concatenate()([color_input, count_input])  # (batch, 4)
        context = layers.Dense((input_shape[0] // 8) * (input_shape[1] // 8) * 1, activation='relu')(context)
        context = layers.Reshape((input_shape[0] // 8, input_shape[1] // 8, 1))(context)

        merged = layers.Concatenate()([bottleneck, context])

        x = layers.UpSampling2D()(merged)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

        output = layers.Conv2D(1, 1, activation='sigmoid')(x)

        model = models.Model(inputs=[image_input, color_input, count_input], outputs=output)
        return model

    def train(self, epochs=20, batch_size=8):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(
            [self.X_train_img, self.X_train_col, self.X_train_count],
            self.Y_train,
            validation_data=(
                [self.X_val_img, self.X_val_col, self.X_val_count],
                self.Y_val
            ),
            batch_size=batch_size,
            epochs=epochs
        )

    def predict(self, X_img=None, X_col=None, X_count=None):
        if X_img is None:
            return self.model.predict([self.images, self.color_features, self.predicted_counts])
        else:
            return self.model.predict([X_img, X_col, X_count])

if __name__ == "__main__":
    unet_trainer = UNetSegmentationTrainer(images=X, masks=Y, predicted_counts=predicted_counts)
    unet_trainer.train(epochs=20)
    pred_masks = unet_trainer.predict()