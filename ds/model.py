import streamlit as st
import pandas as pd
import plotly.express as px
from utils import COMPONENT_NAMES,PROPERTY_NAMES,comp_dict
from page_utils import remove_top


@st.cache_data
def load_data():
     return pd.read_csv("./data/cv_results.csv")

prop_dict = {v: k for k, v in comp_dict.items()}

def model_page():

    remove_top()
    st.set_page_config(
        page_title="Blend Models",
       # page_icon="📚",
        layout="wide",
       # initial_sidebar_state="collapsed"
    )
    with st.spinner():
        df_cv=load_data()
    selection = st.pills(
    "Select one or more property :",
    PROPERTY_NAMES,selection_mode="multi")

    selected_props=[prop_dict[s] for s in selection]
    df_cv_filtered=df_cv[df_cv.property.isin(selected_props)]

    df_cv_filtered.property=df_cv_filtered.property.map(lambda x:comp_dict[x])


    col2,col3 = st.columns([4,4])

  


    with col2:
        fig = px.scatter(
            df_cv_filtered,
            x='y_true',
            y='y_pred',
            color='property',  # Color points by the 'property' column
            title="Scatter Plot of True vs Predicted Values",
            labels={'y_true': 'True Values', 'y_pred': 'Predicted Values'},
            size_max=20  # Maximum size of the markers
        )

        # Update layout if needed
        fig.update_layout(
            legend_title="Property",  # Set the legend title
        )

        # Display the scatter plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col3:
            # Create a box plot using Plotly Express
            fig = px.box(df_cv_filtered, x='property', y='mape', title="Box Plot of Values by Category",color="property")
            # Update the box color to light red (RGB: 255, 200, 200)

            
            # Set y-axis limits
            fig.update_layout(
                yaxis=dict(range=[0, 1]) , # Set the y-axis limits (adjust as needed)
                 legend_title="Property",
                     # Set the legend title
            )

            # Display the box plot in Streamlit
            st.plotly_chart(fig)



model_page()