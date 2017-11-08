# NoReC baseline models

Baselines models trained on the [NoReC dataset](https://github.com/ltgoslo/norec).

## Dependencies

- sklearn
- gensim
- numpy
- scipy

## Installation

To install dependencies, run:

```
$ pip install -r requirements.txt
```

To install package, run:

```
$ pip install .
```

## Usage

Example using `doc2vec` with `regression`:

```
from sant.baselines import load_pipeline

pipeline = load_pipeline("path/to/models/", vectorizer="doc2vec", predictor="regression")

docs = # List of documents, each document being a list of tokens.

predictions = pipeline.predict(docs)
```

The regressor predicts a real-valued number in the approximate range
of 1 to 6, while the classifier predicts an integer from 1 to 6.

## Models

All the models were trained on the training set only. Input was
tokenized and lowercased text.

### Vectorizers / Document representation

- `doc2vec` – `Doc2Vec` from gensim with `iter=55`.
- `bow` – `CountVectorizer` from scikit-learn with the tokenizer and
  analyzer disabled.

### Predictors

- `classification` – `LogisticRegression` from scikit-learn with `C=0.1`.
- `regression` – `Ridge` from scikit-learn with default settings.

## Evaluation

| Model                  | Accuracy | F<sub>1</sub>  | R<sup>2</sup>   |
|------------------------|---------:|---------------:|----------------:|
| bow+classification     |  0.4986  | 0.3537         | 0.1754          |
| bow+regression         |  0.4605  | 0.3186         | 0.2860          |
| doc2vec+classification |  0.4463  | 0.2737         | 0.0708          |
| doc2vec+regression     |  0.4036  | 0.2165         | 0.1402          |

All models were evaluated on the dev set. When calculating accuracy on
the regression models, predictions were rounded and capped to be in
the range 1 to 6.
