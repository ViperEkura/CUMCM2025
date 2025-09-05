from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from scipy.special import logit, expit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class BetaRegression:
    def __init__(self, poly_columns=None, degree=3, include_bias=True):
        self.poly_columns = poly_columns
        self.degree = degree
        self.include_bias = include_bias
        
        # 创建预处理管道
        if poly_columns is not None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('poly', PolynomialFeatures(degree=degree, include_bias=include_bias), poly_columns),
                    ('passthrough', 'passthrough', [i for i in range(len(poly_columns)) if i not in poly_columns])
                ] if isinstance(poly_columns, list) else [('poly', PolynomialFeatures(degree=degree, include_bias=include_bias), poly_columns)]
            )
        else:
            preprocessor = PolynomialFeatures(degree=degree, include_bias=include_bias)
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        self.model = TransformedTargetRegressor(
            regressor=self.pipeline,
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
            'MAE': mean_absolute_error(y, y_pred)
        }