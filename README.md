# Material-discovery

## Set Up

```
conda create --name [env_name] python=3.7
```
```
conda activate [env_name]
```
```
pip install --ignore-installed -r requirements.txt
```
<br/>

## 1. Preprocess the Corpus
```
python preprocessor/preprocess.py --input_file dataset/toyset.txt \
--ouptut_file 01-getCorpus-result.txt \
--output_formula_file 01-getCorpus-result-formula.txt
```
<br/>

## 2. Classify the Abstracts(Corpus)

In advance, you should get [model](https://drive.google.com/file/d/1YwunmwzJ1QlsunJAxeWwr_khMFcofx00/view?usp=drive_link)
and [optimizer](https://drive.google.com/file/d/1H4O9bReCYqrbzpy3T5aTWxnsCUGZDgZ4/view?usp=drive_link).

Notice) Please place the *model* and *optimizer* in the <code>abstract_classifier/inorganic/</code>.
```
cd abstract_classifier/inorganic
```
```
python abst_filter.py --dir_name ../../


# dir_name: Set the location of the .txt file you want to classify.
```
<br/>
    

## 3. Make Phrases
```
cd ../../
```
```
python preprocessor/phraser.py
```
<br/>

## 4. Pretraining the Model
```
python pretraining/03-battery2vec.py --corpus 02-getPhrases-result.txt \
--formulas 01-getCorpus-result-formula.txt \
--model_type word2vec \
--model_name 03-result-model -sg -keep_formula


# model_type: The type of model, which determine how to train.
```
