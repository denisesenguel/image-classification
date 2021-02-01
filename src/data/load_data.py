import tensorflow as tf
import pathlib

# this step can also be done manually
def downloadRGB():

    path = tf.keras.utils.get_file(
        fname=pathlib.Path.cwd() / "data/raw/EuroSAT_RGB/",
        origin="http://madm.dfki.de/files/sentinel/EuroSAT.zip",
        extract=True
    )

    return path

def preprocessImages(path, batch_size, img_height, img_width):

    data_dict = dict()

    for type in ("training", "validation"):

        data_dict[type] = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path,
            subset=type,
            validation_split=0.2,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    return data_dict



