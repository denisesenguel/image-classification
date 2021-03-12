import numpy as np
from tensorflow.keras import layers, Sequential
from src.data.load_data import *
from src.data.transform import normalize_data


def main():

    data_processed = preprocessImages()
    data_normalized = {
        "traning":normalize_data(data_processed["training"]),
        "validation":normalize_data(data_processed["validation"])
    }

    # the following bit is needed for larger testdata

    # AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = data_processed["training"].cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = data_processed["validation"].cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 10

    # model build/definition can go into a separate file
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        data_normalized["training"],
        validation_data=data_normalized["validation"],
        epochs=epochs
    )

    return history

if __name__ == "__main__":
    main()