import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def predict(model, data_processed, path="data/raw/random_forest_testimage.jpg"):

    """
        Classify a new image using trained model
    """

    # find central place for this
    # what if image is of different size than in the training phase?
    img_height = 64
    img_width = 64

    img = keras.preprocessing.image.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(data_processed["validation"].class_names[np.argmax(score)], 100 * np.max(score))
    )

