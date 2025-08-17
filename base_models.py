
import pandas as pd



from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import FunctionTransformer
import numpy as np

from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import random
import numpy as np


from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
import pickle

from lightgbm import LGBMRegressor

#reproducibility !
seed_value=42
np.random.seed(seed_value)
random.seed(seed_value)

class OutputScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None, method=None):
        self.scaler = scaler
        self.method = method

    def fit(self, y, _=None):
        # Fit the scaler on the output
        if self.scaler is not None:
            self.scaler.fit(y)
        return self

    def transform(self, y):
        # Scale the output
        if self.scaler is None:
            return y.reshape(-1, 1)
        if self.method == "log":
            return np.log1p(self.scaler.transform(y.reshape(-1, 1)))
        elif self.method == "inv":
            return 1/(self.scaler.transform(y.reshape(-1, 1)))
        elif self.method == "log10":
            return np.log10(self.scaler.transform(y.reshape(-1, 1)))
        else:
            return self.scaler.transform(y.reshape(-1, 1))

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        # Inverse scale the output
        if self.scaler is None:
            return y.reshape(-1, 1)

        if self.method == "log":
            return self.scaler.inverse_transform(np.expm1(y.reshape(-1, 1)))
        elif self.method == "inv":
            return self.scaler.inverse_transform(1/(y.reshape(-1, 1)))
        elif self.method == "log10":
            return self.scaler.inverse_transform(np.power(10,y.reshape(-1, 1)))
        else:
            return self.scaler.inverse_transform(y.reshape(-1, 1))


def custom_transform(X, features_type=[]):
    """Apply log and inverse transformations to the features."""

    X_transformed = X.copy()
    X_combined = X.copy()
    
    if "log" in features_type:
        X_transformed_log = np.log(X_transformed)
        X_combined = np.concatenate([X_combined, X_transformed_log], axis=1)

    if "log10" in features_type:
        X_transformed_log10 = np.log10(X_transformed)
        X_combined = np.concatenate([X_combined, X_transformed_log10], axis=1)

    if "inv" in features_type:
        X_transformed_inv = 1 / (X_transformed)
        X_combined = np.concatenate([X_combined, X_transformed_inv], axis=1)

    if "is_zero" in features_type:
        X_transformed_nz = pd.DataFrame(np.where(X_transformed[:,:5]==0,1,0))
        X_combined = np.concatenate([X_combined, X_transformed_nz], axis=1)
    
    return pd.DataFrame(X_combined)

def custom_mape(y_true, y_pred,bounds):
    """
    Custom Mean Absolute Percentage Error (MAPE) loss function that minimizes both mean and max MAPE.
    """
    y_pred_clipped = np.where((y_pred >= bounds[0]) & (y_pred <= bounds[1]), 0.0, y_pred)
    mape = np.abs(y_true - y_pred_clipped) / np.abs(y_true)
    
    return mape

def get_sample_weight(y,alpha=0):
    sample_weight=(np.exp(-(y)**2*alpha))
    sample_weight/=sum(sample_weight)
    return sample_weight.ravel()

def get_score(cost):
    return max(10,100-(90*cost)/2.72)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / abs(y_true)))

def train_model(target_name,models_dict,df_train,X_sub):

    output_scaler=OutputScaler(scaler=MinMaxScaler((1,2)),method=models_dict[target_name][0])

    transformer = FunctionTransformer(func=custom_transform, kw_args={"features_type":models_dict[target_name][2]},validate=False)

    X = df_train[[c for c in df_train.columns if "blend" not in c]]
    y = output_scaler.fit_transform(df_train[target_name].values.reshape(-1,1))

    model = models_dict[target_name][1]

    # Define a pipeline (optional, if you have preprocessing steps)
    pipeline = Pipeline([
        ('scaler',MinMaxScaler((1,2))),
        ('custom_transform', transformer),
        ('poly_features', PolynomialFeatures(2,interaction_only=models_dict[target_name][3])),  # Optional: Scale features
        ('model', model)
    ])

    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    scores = []

    for train_index, test_index in rs.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sample_weight=get_sample_weight(y_train)

        pipeline.fit(X_train, y_train.ravel(),model__sample_weight=sample_weight)
        
        y_pred = pipeline.predict(X_test)
        mape = custom_mape(output_scaler.inverse_transform(y_test), output_scaler.inverse_transform(y_pred),models_dict[target_name][4])
        min_score,median_score,mean_score,max_score=get_score(np.quantile(mape,0.95)),get_score(np.median(mape)),get_score(np.mean(mape)),get_score(np.min(mape))
        
        l_scores=[int(min_score),int(median_score),int(mean_score),int(max_score)]
        scores.append(l_scores)
        print(l_scores)

    #train on all dataset and export pred submission

    sample_weight=get_sample_weight(y)
    pipeline.fit(X, y.ravel(),model__sample_weight=sample_weight)

    #save model
    with open(f'./models/best_model_{target_name}.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    y_sub_pred = output_scaler.inverse_transform(pipeline.predict(X_sub))

    bounds=models_dict[target_name][4]
    y_sub_pred_clipped = np.where((y_sub_pred >= bounds[0]) & (y_sub_pred <= bounds[1]), 0.0, y_sub_pred)

    pd.Series(y_sub_pred_clipped.ravel()).to_csv(f"../submission/{target_name}_{model.__class__.__name__}_{mean_score:.2f}.csv")

    return scores