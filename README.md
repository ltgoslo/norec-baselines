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

## Comments to the evaluation results

Note that the parameter tuning of the classifiers and regression models are
optimised with respect to accuracy and R<sup>2</sup>, respectively. We evaluate all
models with respect to both metrics however, in addition to macro-averaged F<sub>1</sub>.

The results let us make at least three important observations: (1) The BoW
representations give better results than doc2vec. This is likely because the
relevant cues for predicting polarity are local properties in the
documents. While doc2vec can be seen to average the contribution of all tokens
in the document, the BoW representation allows the model to learn different
weights for different words. This is especially important given that we work
with relatively long documents (approx. 420 tokens on average).  (2) We see
that the models tend to perform best relative to the metric they where
optimized for. However, we would argue that the regression approach, which has
the strongest performance with respect to R<sup>2</sup>, would be the most promising
direction to pursue.  While accuracy (and the classification-based models)
treats all ratings as equally distinct, the R<sup>2</sup> metric takes into account
the property that the ratings 2 and 3 are closer than 2 and 6. (3) There is
still ample room for improvement, however.  The performance metrics clearly
show that these baselines results are exactly that; preliminary results that
tell us something about the difficulty of the task, providing a point of
reference to be improved upon.
