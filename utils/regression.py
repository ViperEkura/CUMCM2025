from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import logit, expit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class BetaRegression(BaseEstimator, RegressorMixin):
    def __init__(self, poly_columns=None, degree=3, include_bias=True):
        self.poly_columns = poly_columns
        self.degree = degree
        self.include_bias = include_bias
        self.model = None

    def _build_pipeline(self, n_features):
        """根据特征数量构建管道"""
        if self.poly_columns is not None:
            if not isinstance(self.poly_columns, list):
                poly_columns = [self.poly_columns]
            else:
                poly_columns = self.poly_columns
            
            transformers = [
                ('poly', PolynomialFeatures(degree=self.degree, include_bias=self.include_bias), poly_columns)
            ]
            
            # 添加非多项式列的处理
            non_poly_columns = [i for i in range(n_features) if i not in poly_columns]
            if non_poly_columns:
                transformers.append(('passthrough', 'passthrough', non_poly_columns))
            
            preprocessor = ColumnTransformer(transformers=transformers)
        else:
            preprocessor = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

    def fit(self, X, y):
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        pipeline = self._build_pipeline(n_features)
        
        self.model = TransformedTargetRegressor(
            regressor=pipeline,
            func=logit,
            inverse_func=expit
        )
        
        self.model.fit(X, y)
        return self

    def predict(self, X):
        assert self.model is not None
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'R2': r2_score(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred)
        }