from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from scipy.special import logit, expit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  

class BetaRegression:
    def __init__(self, degree=3, include_bias=True):
        self.pipeline = Pipeline([
            ('poly_features', PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        self.model = TransformedTargetRegressor(
            regressor=self.pipeline,
            func=logit,
            inverse_func=expit
        )
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    # 新增评估方法
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return {
            'R2': r2,
            'MSE': mse,
            'MAE': mae
        }