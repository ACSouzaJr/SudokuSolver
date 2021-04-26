# Usage
# train_number_classifier.py --output output/number_classifier.h5

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from models.SudokuModel import SudokuModel
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

EPOCHS = 10
BATCH_SIZE = 250

if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Path to output model after training")

    args = vars(parser.parse_args())

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Pre-process Data
    ## Normalization - scale [0.01, 1]
    X_train = X_train / 255.0 * 0.99 + 0.01
    X_test = X_test / 255.0 * 0.99 + 0.01

    ## Reshape
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    ## Label encoding - Convert to one-hot-encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Split Train, Validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=1)

    # Create model
    model = SudokuModel.build()

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    ## Data Augumentation - Create image variations
    datagen = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    ## Early Stopping (callback)
    early_stopping = EarlyStopping(
        mode="min",
        patience=6,  # how many epochs to wait before stopping
        restore_best_weights=True,
        verbose=1
    )

    # Fit model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
        callbacks=[early_stopping]
    )

    # Plot Graph
    history_df = pd.DataFrame(history.history)

    history_df.loc[:, ['loss', 'val_loss']].plot()
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot()

    print(("Best Validation Loss: {:0.4f}" + \
           "\nBest Validation Accuracy: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_accuracy'].max()))

    # Evaluate the network
    print("[INFO] evaluating network...")

    predictions = model.predict(X_test)

    print(classification_report(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in range(10)])
    )

    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save(args["output"], save_format="h5")
