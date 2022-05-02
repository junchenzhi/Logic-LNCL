# Logic-LNCL
Implementation of the Logic-LNCL on Sentiment Polarity (MTurk) Dataset and CoNLL-2003 NER (MTurk) dataset.

## 1 Sentiment Polarity (MTurk) Dataset
### 1.1 Prepare 
- Download pre-trained word embeddings [GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download ''), place it in `./sentiment/data/`

### 1.2 Denpendencies: 
`conda env create -f sentiment.yaml`
####  Or configure:
- python 3.6.12
- pytorch 1.7.1
- gensim 3.8.0
- numpy 1.19.2
- scikit-learn 0.23.2 

### 1.3 Training/Evaluation
- `python main.py --result_path Logic-LNCl_1`
- Write a `*.sh` script to run multiple times, e.g., `run.sh`



## 2 CoNLL-2003 NER (MTurk) dataset
### 1.1 Prepare 
- Download pre-trained word embeddings with [glove.6B.300D.txt](https://nlp.stanford.edu/projects/glove/ ''), place it in `./NER/data/`

### 1.2 Denpendencies: 
`conda env create -f ner.yaml`
####  Or configure:
- python 3.5.6
- tensorflow 1.10.0
- keras 2.2.2
- numpy 1.15.2
- scikit-learn 0.20.2 

### 1.3 Training/Evaluation
- `python main.py Logic-LNCl_1`
- Write a `*.sh` script to run multiple times, e.g., `run.sh`