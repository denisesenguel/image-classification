import tensorflow as tf

def normalize_data(data):

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = data.map(lambda x, y: (normalization_layer(x), y))

    # image_batch, labels_batch = next(iter(normalized_ds))

    return normalized_ds