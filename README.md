Personal MLops exercises repository
==============================

repo to showcase my work during the DTU MLOps course winter 2022 session

## Running the project

Assuming the files from [week 1](https://github.com/SkafteNicki/dtu_mlops/tree/main/data/corruptmnist) are aggregated in `data/processed/train.npz` and `data/processed/test.npz` as `train_merged.npz` and `test.npz`, you can run the following:

training loop:  

    make train  

will train the model and save it in `models/cnn_classifier` or `mnist_classifier`
depending on the arguments in the config file [todo]

if a model is saved in those locations, running the evaluation loop:

```    make evaluate ```

you can also perform a sanity check on data and model by running
<br>
```  pytest            ```<br>
<br>

will evaluate this models accuracy.<br>
Things like hyperparamters, save locations, can be modified in the file `src/config/config_CNN.yaml`

## Things I gone and done that work

- [x] Working mnist classifier, one CNN (91% acc) and one regular NN (92% acc)
- [x] did some pep8 compliance at some point (not anymore haha)
- [x] cookie-cutter structure
- [x] Makefile
- [x] configuration loading and producing logs with hydra
- [x] unit test exercises, though coverage is still low
- [x] Docker
- [x] cloud-build check with a docker image
- [x] ran in a VM on gcp but this wasn't very interesting
- [x] learned a bunch of stuff

## Things I tried but didn't succeed because of lack of time and/or skills :(

- [] setting up cloud functions
- [] inference
- [] profiling
- [] pytorch-lightning training loop


Project Organization. Loosely based on cookie cutter structure, but I didn't respect it all the way
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- data I feed to my models after running some stuff on raw
    │   └── raw            <- the content from week1 - not used anymore
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. (empty)
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt` or `pipreqs`
    │
    ├── Dockerfile         
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├──load_mnist.py <- produces dataloaders from data in /data/processed
    │   │   └──
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented 
    │
    ├── tests
    │   ├── __init__.py    <- Makes tests a Python module
    │   ├── test_model.py  <- tests pertaining to model
    │   └── test_data.py   <- tests pertaining to data
    │
    ├──config              <- contains config file to load hyperparameters and other commands using hydra
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
