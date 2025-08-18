import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils import COMPONENT_NAMES,PROPERTY_NAMES,comp_dict
from page_utils import remove_top

remove_top()
st.set_page_config(
    page_title="Explainability",
    # page_icon="📚",
    layout="wide",
    #initial_sidebar_state="collapsed"
)
comp_dict_new={
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
"property1":"Density",
"property2":"Viscosity",
"property3":"Heating Value",
"property4":"Cetane Number",
"property5":"Freezing Point",
"property6":"Smoke Point",
"property7":"Thermal Stability",
"property8":"Water Content",
"property9":"Particulate Matter",
"property10":"Corrosiveness",
"component1":"Diesel",
"component2":"Biofuel",
"component3":"Synthetic Fuel",
"component4":"Additives",
"component5":"Waste Oil",
}

selection = st.pills(
    "Select one property:",
    PROPERTY_NAMES,selection_mode="single",default=PROPERTY_NAMES[0])

# Load your dataset
shap_df = pd.read_csv('data/shap_values.csv')  # Assuming these are the relevant features
shap_df["model_name"]=shap_df["model_name"].map(lambda x:comp_dict[x.lower()])



shap_df_f=shap_df[shap_df.model_name==selection].drop(columns=["model_name"]).dropna(axis=1, how='all')

# Function to replace values based on the dictionary
def replace_column_names(col_name):
    # Split the column name by underscore if it contains multiple items
    new_name=col_name.lower()
    print(new_name)
    for k,v in reversed(comp_dict_new.items()):
        new_name=new_name.replace(k,v)
    return new_name

# Apply the function to the DataFrame column
shap_df_f.columns = [replace_column_names(col) for col in shap_df_f.columns]

# Create a Plotly box plot for the top features
box_fig = go.Figure()

for feature in shap_df_f.columns:
    box_fig.add_trace(go.Box(y=shap_df_f[feature], 
                             name=feature,
                            marker=dict(color='teal'),
                            line=dict(color='teal'),
                             )
                    )
box_fig.update_layout(showlegend=False)
# Streamlit app layout
st.plotly_chart(box_fig)
