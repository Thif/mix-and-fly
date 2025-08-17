import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import streamlit as st
import tabular_transformer as tm
import base_models as bm
import joblib




comp_dict={
"blendproperty1":"Density",
"blendproperty2":"Viscosity",
"blendproperty3":"Heating Value",
"blendproperty4":"Cetane Number",
"blendproperty5":"Freezing Point",
"blendproperty6":"Smoke Point",
"blendproperty7":"Thermal Stability",
"blendproperty8":"Water Content",
"blendproperty9":"Particulate Matter",
"blendproperty10":"Corrosiveness",
"component1_fraction":"Diesel Fraction",
"component2_fraction":"Biofuel Fraction",
"component3_fraction":"Synthetic Fuel Fraction",
"component4_fraction":"Additives Fraction",
"component5_fraction":"Waste Oil Fraction",
}


compliance_criteria = {
    "Density": (0, 4.0),  # Example range for normalized density
    "Viscosity": (0, 4),  # Example range for normalized viscosity
    "Heating Value": (0, 4.0),  # Example range for normalized heating value
    "Cetane Number": (0, 4.0),  # Example range for cetane number
    "Freezing Point": (-4.0, 0.0),  # Example range for freezing point
    "Smoke Point": (0.0, 4.0),  # Example range for smoke point
    "Thermal Stability": (0.0, 4.0),  # Example range for thermal stability
    "Water Content": (0.0, 4.0),  # Example range for water content
    "Particulate Matter": (0.0, 4.0),  # Example range for particulate matter
    "Corrosiveness": (0.0, 4.0),  # Example range for corrosiveness
    "Diesel Fraction": (0.0, 4.0),  # Example range for diesel fraction
    "Biofuel Fraction": (0.0, 4.0),  # Example range for biofuel fraction
    "Synthetic Fuel Fraction": (0.0, 1.0),  # Example range for synthetic fuel fraction
    "Additives Fraction": (0.0, 1.0),  # Example range for additives fraction
    "Waste Oil Fraction": (0.0, 1.0),  # Example range for waste oil fraction
}

COMP_COST=[3,5,6,0.5,0]
PROPERTY_NAMES=[v for (k,v) in comp_dict.items() if "Fraction" not in v]
PREDICTED_PROPERTY_NAMES=[v+" (Predicted)" for v in PROPERTY_NAMES]
ORIGINAL_COMPONENT_NAMES=[k for (k,v) in comp_dict.items() if "Fraction" in v]
COMPONENT_NAMES=[v for (k,v) in comp_dict.items() if "Fraction" in v]
METRIC_NAMES=["Safety","Performance","Sustainability","Cost","Compliant"]

@st.cache_resource
def load_tf():
    import tensorflow as tf
    return tf

@st.cache_resource
def load_model_func(model_path):
    tf=load_tf()
    if "keras" in model_path:
        return tf.keras.models.load_model(model_path)
    else:
        with open(model_path, 'rb') as file:
            return pickle.load(file)

@st.cache_data
def read_dataset(path):
    return pd.read_csv(path)

def check_compliance(row):
    for property_name, (min_threshold, max_threshold) in compliance_criteria.items():
        if not (min_threshold <= row[property_name] <= max_threshold):
            return False
    return True


def generate_random_fractions(alpha=[1, 1, 1, 1, 1], num_samples=10):
    valid_samples = []
    lower_bounds = np.array([0.0, 0.0, 0.0, 0.05, 0.0])
    upper_bounds = np.array([0.5, 0.5, 0.5, 0.5, 0.29])
    while len(valid_samples) < num_samples:
        # Generate a sample from the Dirichlet distribution
        sample = np.random.dirichlet(alpha)
        
        # Check if all values in the sample are less than or equal to 0.5
        if np.all(sample > lower_bounds) & np.all(sample < upper_bounds):
            valid_samples.append(sample)
    
    return np.array(valid_samples)

def calculate_metrics(X,X_comp):
    X.columns=PROPERTY_NAMES
    scaler=MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    x_comp_scaled=pd.DataFrame(scaler.fit_transform(X_comp), columns=X_comp.columns)

    performance = np.average(pd.concat([
        df_scaled["Heating Value"],
        df_scaled["Cetane Number"],
        (1 - df_scaled["Viscosity"]),
        x_comp_scaled["Diesel Fraction"]
    ], axis=1),weights=[1,1,1,5],axis=1)  # Calculate the mean across the rows

    safety=np.average(pd.concat([
    (1 - df_scaled["Freezing Point"]),
    (1 - df_scaled["Thermal Stability"])
    ],axis=1),weights=[1,5],axis=1)

    sustainability=np.average(pd.concat([(1-df_scaled["Water Content"]),
                            (1-df_scaled["Particulate Matter"]),
                            (x_comp_scaled["Biofuel Fraction"]),
                            (x_comp_scaled["Synthetic Fuel Fraction"])],axis=1),weights=[1,1,5,1],axis=1)
        
    cost=X_comp[COMPONENT_NAMES].multiply(COMP_COST).sum(axis=1)
    compliant = pd.concat([X,X_comp],axis=1).apply(check_compliance, axis=1)

    return np.round(performance,2),np.round(safety,2),np.round(sustainability,2),np.round(cost,2),np.round(compliant,2)

def get_predictions_for_blend(X):

    X=X.drop(columns=[c for c in X if "blendproperty" in c])
    if "ID" in X.columns:
        X=X.drop(columns=["ID"])

    with open('./models/best_model_blendproperty1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    pred=loaded_model.predict(X)
    return pred

@st.cache_resource
class BlendDataset():
    def __init__(self,path="./data/webapp_data.csv"):
        self.path=path

    def load_data(self):
        df=read_dataset(self.path)
        df.columns = df.columns.str.lower()
        self.input_columns=[c for c in df.columns if "blendproperty" not in c]
        self.target_columns=[c for c in df.columns if "blendproperty" in c]
        self.original_features_df=df[self.input_columns].copy()
        self.original_target_df=df[self.target_columns].copy()
        df["ID"]=df.index+1
        return df

    def rename_cols(self,df):
        df=df.rename(columns=comp_dict)
        return df

    def add_metrics(self,df,predicted=False):
        if predicted:
            df["Performance"],df["Safety"],df["Sustainability"],df["Cost"],df["Compliant"]=calculate_metrics(df[PREDICTED_PROPERTY_NAMES],df[COMPONENT_NAMES])
        else:
            df["Performance"],df["Safety"],df["Sustainability"],df["Cost"],df["Compliant"]=calculate_metrics(df[PROPERTY_NAMES],df[COMPONENT_NAMES])

        return df


    def add_properties(self, df):

        df=df.copy()
        tf=load_tf()


        # Dictionary mapping property names to their corresponding model file paths
        model_dict = {
            "Density (Predicted)": './models/best_model_blendproperty1.pkl',
            "Viscosity (Predicted)": './models/best_model_BlendProperty2.keras',
            "Heating Value (Predicted)": './models/best_model_BlendProperty3.keras',
            "Cetane Number (Predicted)": './models/best_model_blendproperty4.pkl',
            "Freezing Point (Predicted)": './models/best_model_blendproperty5.pkl',
            "Smoke Point (Predicted)": './models/best_model_BlendProperty6.pkl',
             "Thermal Stability (Predicted)": './models/best_model_BlendProperty7.keras',
              "Water Content (Predicted)": './models/best_model_BlendProperty8.keras',
               "Particulate Matter (Predicted)": './models/best_model_BlendProperty9.keras',
            "Corrosiveness (Predicted)": './models/best_model_blendproperty10.pkl'
        }

        # Loop through the dictionary to load models and make predictions
        for property_name, model_path in model_dict.items():
            try:
                loaded_model=load_model_func(model_path)
                if ".keras" in model_path:              
                    #scale inputs
                    scaler_input = joblib.load('models/blendproperty2_scaler_input.pkl')
                    inputs=scaler_input.transform(df[self.input_columns]).reshape(len(df[self.input_columns]),11,5)
                    inputs = tf.convert_to_tensor(inputs)
                    df.loc[:,property_name] = loaded_model.predict(inputs)
                else:
                    inputs=df[self.input_columns]

                    if property_name!="Cetane Number (Predicted)":
                        output_scaler=bm.OutputScaler(scaler=MinMaxScaler((1,2)),method=None)
                    else:
                        output_scaler=bm.OutputScaler(scaler=MinMaxScaler((1,2)),method="log")

                    original_df_new_names=self.original_target_df.copy()
                    original_df_new_names.columns=PREDICTED_PROPERTY_NAMES
                    _ = output_scaler.fit_transform(original_df_new_names[property_name].values.reshape(-1,1))
                    df.loc[:,property_name] = output_scaler.inverse_transform(loaded_model.predict(inputs))
                      
            except Exception as e:
                print(e)
                df[property_name]=-1
        return df

    def generate_samples_from_id(self,df,id,num_samples=100):
        df_single_row=df[df.ID==id].drop(columns=COMPONENT_NAMES)
        df_duplicates = pd.concat([df_single_row]*num_samples, ignore_index=True)
        df_fractions=pd.DataFrame(generate_random_fractions(num_samples=num_samples),columns=ORIGINAL_COMPONENT_NAMES)
        return pd.concat([df_fractions,df_duplicates],axis=1)


