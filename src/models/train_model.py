import numpy as np
from src.data.load_data import *


def main():

    data_processed = preprocessImages()

    # the following bit is needed for larger testdata

    # AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = data_processed["training"].cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = data_processed["validation"].cache().prefetch(buffer_size=AUTOTUNE)

    data_split =



    np.random.seed(108)




if __name__ == "__main__":
    main()