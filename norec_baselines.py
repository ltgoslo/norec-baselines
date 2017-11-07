# Copyright 2017 Eivind Alexander Bergem <eivinabe@ifi.uio.no>
# This file is part of norec.

# norec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# norec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with norec.  If not, see <http://www.gnu.org/licenses/>.

import os.path

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from gensim.models import doc2vec

# Hack to fix namespace issues when unpickling
import __main__
__main__.identity = lambda x : x

MODELS = {"doc2vec": "doc2vec.model.gz",
          "bow": "CountVectorizer.model.xz",
          "regression": "Ridge.model.xz",
          "classification": "LogisticRegression.model.xz"}

class Doc2VecWrapper(object):
    """SKlearn wrapper around doc2vec."""

    def __init__(self, filename):
        self.d2v = doc2vec.Doc2Vec.load(filename)

    def transform(self, X):
        """Transform X and return."""

        # Create numpy array to hold output
        m = len(X)
        n = self.d2v.vector_size
        out = np.ndarray((m, n))

        # Infer vector for each document
        for i in range(m):
            out[i] = self.d2v.infer_vector(X[i])

        return out

    # Add placeholder to supress Pipeline error
    def fit(self, X):
        pass

def load_vectorizer(path, name):
    """Load vectorizer."""

    filename = os.path.join(path, name, MODELS[name])

    print("Loading vectorizer: {}".format(filename))
    if name == "doc2vec":
        return Doc2VecWrapper(filename)
    else:
        return joblib.load(filename)

def load_predictor(path, vectorizer, predictor):
    """Load predictor."""

    filename = os.path.join(path, vectorizer, MODELS[predictor])

    print("Loading vectorizer: {}".format(filename))
    return joblib.load(filename)

def load_pipeline(models_path, vectorizer="bow", predictor="regression"):
    """Load pipeline."""

    print("Loading pipeline '{}+{}'".format(vectorizer, predictor))
    return Pipeline([('vect', load_vectorizer(models_path, vectorizer)),
                     ('pred', load_predictor(models_path, vectorizer,
                                             predictor))])
