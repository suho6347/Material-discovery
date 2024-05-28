# Material-discovery

## Set Up

    conda create --name [env_name] python=3.7

    conda activate [env_name]
    
    pip install --ignore-installed -r requirements.txt


## How to preprocessing your corpus,
toyset usage

    python preprocessor/preprocess.py

## How to classify abstract
toyset usage

need to get [model](https://drive.google.com/file/d/1YwunmwzJ1QlsunJAxeWwr_khMFcofx00/view?usp=drive_link)

need to get [optimizer](https://drive.google.com/file/d/1H4O9bReCYqrbzpy3T5aTWxnsCUGZDgZ4/view?usp=drive_link)

Please place the *model* and *optimizer* in the **abstract_classifier/inotganic** folder.

    cd abstract_classifier/inorganic

    python abst_filter.py --dir_name ../../

argument description as follow:

    --dir_name : Set the location of the .txt file you want to classify.

## How to make phrases
toyset usage

    cd ../../

    python preprocessor/phraser.py

## How to pretrain model
toyset usage

    python pretraining/03-battery2vec.py --corpus 02-getPhrases-result.txt --formulas 01-getCorpus-result-formula.txt --model_name 03-result-model -sg -keep_formula

argument description as follow:

    --model_type : The type of model, which determine how to train. The options are word2vec or fasttext.
