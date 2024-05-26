# Material-discovery

how to preprocessing your corpus,
    toyset usage
    `python preprocessor/preprocess.py`

how to classify abstract
    toyset usage
    `cd abstract_classifier/ingorganic`
    `python abst_filter.py --dir_name [dir_path]`

how to make phrases
    toyset usage
    `python preprocessor/phraser.py`

how to pretrain model
    toyset usage
    `python pretraining/03-battery2vec.py --corpus 02-getPhrases-result.txt --formulas 01-getCorpus-result-formula.txt --model_name 03-result-model -sg -keep_formula`