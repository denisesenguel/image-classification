import numpy as np
from src.data.load_data import *


def main():

    data_split = preprocessImages()["training"]

    np.random.seed(108)




if __name__ == "__main__":
    main()