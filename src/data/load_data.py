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

def preprocessImages(batch_size=100, img_height=64, img_width=64):

    path = downloadRGB()

    data_dict = dict()

    for type in ("training", "validation"):

        # set seed for split?
        data_dict[type] = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path,
            subset=type,
            validation_split=0.2,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    return data_dict



