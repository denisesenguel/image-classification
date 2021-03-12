import tensorflow as tf
import pathlib
import zipfile


# this step can also be done manually
def downloadRGB():

    path_to_zip = tf.keras.utils.get_file(
        fname=pathlib.Path.cwd() / "data/raw/EuroSAT_RGB.zip",
        origin="http://madm.dfki.de/files/sentinel/EuroSAT.zip",
        archive_format="zip",
        extract=True
    )

    # somehow the above fcn doesn't extract, seems to be a bug
    # unzip manually
    # this does not work properly yet
    path_extracted = pathlib.Path(path_to_zip).parent / "/EuroSAT_RGB/"
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(path_extracted)

    return path_extracted


def preprocessImages(batch_size=100, img_height=64, img_width=64):

    path = pathlib.Path.cwd() / "data/raw/EuroSAT_RGB/"

    # download only needs to be done once on every machine.
    # path = downloadRGB()

    data_dict = dict()

    for type in ("training", "validation"):

        # set seed for split?
        data_dict[type] = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path,
            labels="inferred",
            subset=type,
            validation_split=0.2,
            seed=149,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    return data_dict



