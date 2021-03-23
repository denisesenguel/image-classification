from tensorflow.keras import layers, Sequential
from src.data.load_data import *
from src.data.transform import normalize_data
import time

def main():

    img_height = 64
    img_width = 64

    data_processed = preprocessImages(img_width=img_width, img_height=img_height)
    data_normalized = {
        "training":normalize_data(data_processed["training"]),
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

    epochs = 20

    # use keras callbacks to get time per epoch, see bookmark in SO
    begin = time.process_time()
    history = model.fit(
        data_normalized["training"],
        validation_data=data_normalized["validation"],
        epochs=epochs
    )
    elapsed = time.process_time() - begin
    print("total time elapsed during model fitting: " + str(round(elapsed, 3)))

    

    return history

if __name__ == "__main__":
    main()