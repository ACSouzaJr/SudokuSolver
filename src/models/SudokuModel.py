from tensorflow import keras
from tensorflow.keras import layers


class SudokuModel:
    @classmethod
    def build(cls, width=28, height=28, depth=1, classes=10):
        model = keras.Sequential([
            layers.InputLayer((width, height, depth)),

            # First convolutional block: CONV2D -> Activation -> Maxpooling
            layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=5, strides=2, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Second convolutional block
            layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=5, strides=2, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Classifier Head -> Dense Layers
            layers.Flatten(),

            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(classes, activation="softmax")
        ])

        return model
