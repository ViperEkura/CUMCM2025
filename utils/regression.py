from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import logit, expit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class BetaRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        pipeline = Pipeline([
            ('regressor', LinearRegression())
        ])
        
        self.model = TransformedTargetRegressor(
            regressor=pipeline,
            func=logit,
            inverse_func=expit
        )

    def fit(self, X, y):        
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'R2': r2_score(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred),
            "y_pred": y_pred
        }
        
    def get_coefficients(self):
        pipeline = self.model.regressor_
        linear_model = pipeline.named_steps['regressor']
        
        coef = linear_model.coef_
        intercept = linear_model.intercept_
        n_features_in = coef.size
        feature_names = [f"x{i}" for i in range(n_features_in)]

        return coef, intercept, feature_names