from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ConcatenatingTransformer(BaseEstimator, TransformerMixin):
    """
    Applies multiple transformers to the same input and concatenates
    their outputs along the feature axis.

    Args:
        transformers: list of (name, transformer) tuples, following
                      sklearn's FeatureUnion convention.
    """

    def __init__(self, transformers: list[tuple[str, BaseEstimator]]):
        self.transformers = transformers

    def fit(self, X, y=None, **fit_params):
        for name, transformer in self.transformers:
            params = {
                k[len(name) + 2:]: v
                for k, v in fit_params.items()
                if k.startswith(f"{name}__")
            }
            transformer.fit(X, y, **params)
        return self

    def transform(self, X, **transform_params):
        parts = []
        for name, transformer in self.transformers:
            params = {
                k[len(name) + 2:]: v
                for k, v in transform_params.items()
                if k.startswith(f"{name}__")
            }
            parts.append(transformer.transform(X, **params))
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, **{
            k: v for k, v in fit_params.items()
        })

    def predict(self, X, **predict_params):
        return self.transform(X, **predict_params)