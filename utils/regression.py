from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from scipy.special import logit, expit


class BetaRegression:
    def __init__(self):
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        self.model = TransformedTargetRegressor(
            regressor=pipeline,
            func=logit,
            inverse_func=expit
        )
        

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)