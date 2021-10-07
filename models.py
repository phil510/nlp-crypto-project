import numpy as np
from itertools import product
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model


def roll_array(arr, lags=1):
    """Convert one dimensional array into matrix with lagged values."""
    x = np.stack([np.roll(arr, l) for l in range(lags + 1)])
    x = x[1:, lags:].T
    
    return x


class ARModel:
    """
    Effectively a wrapper around the statsmodels AutoReg class to make
    in sample and out of sample predictions easier
    """
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        
        # get a few of the values we'll need for predictions
        self.lags = self.init_kwargs['lags']  # required kwarg
        self.trend = self.init_kwargs.get('trend', 'c')  # optional kwarg
        
        # make sure the trend and lag values are consistent with the
        # predict function implementation below 
        assert isinstance(self.lags, int), 'List value for lags is not currently supported.'
        assert ('t' not in self.trend), 'Time trend models are not currently supported.'

    def fit(self, y=None, x=None, **kwargs):
        self.model = AutoReg(y, exog=x, **self.init_kwargs)
        self.res = self.model.fit(**kwargs)

    def in_sample_predict(self):
        # predict will provide a prediction starts at t1, not t0,
        # as the t1 prediction depends on the t0 value
        prediction = self.res.predict(start=0, end=-1)
        assert (self.res.nobs == len(prediction))

        return prediction

    def predict(self, y=None, x=None):
        y_lagged = roll_array(y, self.lags)
        if self.trend == 'c':
            y_lagged = sm.add_constant(y_lagged)

        if x is not None:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x = x[self.lags:]
            x = np.concatenate([y_lagged, x], axis=1)
        else:
            x = y_lagged

        prediction = x @ self.res.params

        return prediction


class GARCHModel:
    """
    Effectively a wrapper around the arch arch_model class to make
    in sample and out of sample predictions easier
    """
    def __init__(self, scale=True, **kwargs):
        self.scale = scale
        self.init_kwargs = kwargs

    def fit(self, y=None, x=None, **kwargs):
        if self.scale:
            y = y * 100
        self.model = arch_model(y, **self.init_kwargs)
        self.res = self.model.fit(**kwargs)

    def in_sample_predict(self):
        # first value is nan as the prediction at t1 is based on the value of t0
        # to get a value for t0, use res.conditional_volatility
        prediction = self.res.forecast(start=0, reindex=False, align='target').variance[1:]
        assert (self.res.nobs - 1 == len(prediction))
        
        # convert to volatility and scale if needed
        prediction = np.sqrt(prediction)
        if self.scale:
            prediction = prediction / 100

        return prediction

    def predict(self, y=None, x=None):
        # NOTE: predict does NOT provide a prediction for the first value of endog
        # initiate a temporary model with the new data and
        # set the parameters of the new model to the fit parameters
        if self.scale:
            y = y * 100
        temp_model = arch_model(y, **self.init_kwargs)
        temp_res = temp_model.fix(self.res.params)

        # predict using the new temp model results
        prediction = temp_res.forecast(start=0, reindex=False, align='target').variance[1:]
        prediction = prediction.values.flatten()
        assert (temp_res.nobs - 1 == len(prediction))

        # convert to volatility and scale if needed
        prediction = np.sqrt(prediction)
        if self.scale:
            prediction = prediction / 100

        return prediction


class GridSearch:
    """
    Grid search class that is compatible with the ARModel, GARCHModel
    and EWMAModels herein.
    """
    def __init__(self, model_class, params, eval_fn, 
                  higher_is_better=True):   
        self.model_class = model_class
        self.eval_fn = eval_fn
        self.higher_is_better = higher_is_better
        self.create_grid(params)

        self.trained_models = None
        self.scores = None
        self.best_params = None

    def create_grid(self, params):
        keys, values = zip(*params.items())
        self.param_grid = []
        for v in product(*values):
            params_set = dict(zip(keys, v))
            self.param_grid.append(params_set)

    def search(self, y_train=None, x_train=None, x_val=None, y_val=None,
               **kwargs):
        best_score = -np.inf if self.higher_is_better else np.inf
        self.trained_models = []
        self.scores = []

        for params in self.param_grid:
            model = self.model_class(**params)
            model.fit(y=y_train, x=x_train, **kwargs)
            self.trained_models.append(model)

            score = self.eval_fn(model, y=y_val, x=x_val)
            self.scores.append(score)

            if self.higher_is_better:
                if score > best_score:
                    self.best_params = params
                    best_score = score
            else:
                if score < best_score:
                    self.best_params = params
                    best_score = score

        return self.best_params, best_score


class EWMAModel:
    """
    Simple wrapper around the pandas ewm functionality.
    Setup like this so we can use GridSearch to find the 
    best value of alpha.
    """
    def __init__(self, alpha=None):
        self.alpha = alpha

    def fit(self, y=None, x=None, **kwargs):
        pass

    def predict(self, y=None, x=None):
        return x.ewm(alpha=self.alpha).mean()