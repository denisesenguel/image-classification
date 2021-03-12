image-classification
==============================

Building a first base convolutional neural network model for land use classification based on 
(EUROSAT Sentinel2 satellite imagery)[https://github.com/phelber/eurosat]. Available Classes are..
- Industrial Buildings
- Residential Buildings
- Annual Crop
- Permanent Crop
- River
- Sea & Lake
- Herbaceous Vegetation
- Highway
- Pasture (Weide)
- Forest
The dataset in total contains 27.000 geo-referenced and classified images.

# Local Setup

First-time use
- Make sure you have Python 3 and Anaconda installed and configured
- run `conda env create -f environment.yml`

If you've already done so, simply run `conda activate tf`

Currently the code relies on the `tensorflow.keras` function `image_dataset_from_directory` which requires Tensorflow 
version 2.3.0 or higher, which is currently not yet supported by Anaconda. 
For now ignore any conda envs and use your local Python3 installation addi the most recent Tensorflow version by 
simply running `pip3 install tensorflow`. If Anaconda does not support 2.3.+ soon we can still switch to standard pip envs.

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── cleaned        <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml    <- The environment configs for reproducing the analysis environment. Re-create or update by running `conda env export > environment.yml
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, generate or pre-process data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create results oriented visualizations

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
