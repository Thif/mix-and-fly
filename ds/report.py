import streamlit as st
import pandas as pd
import plotly.express as px
from utils import COMPONENT_NAMES,PROPERTY_NAMES,comp_dict
from page_utils import remove_top


def report_page():

    st.set_page_config(
        page_title="Add blend",
        #page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    remove_top()

    st.markdown(
        """
        <h1 style="text-align: center;">🚧 Page under construction 🚧</h1>
        """,
        unsafe_allow_html=True
    )
        
    # File uploader for PDF report
    pdf_file = st.file_uploader("Upload PDF Report", type=["pdf"])
    
    if pdf_file is not None:
        # Display the uploaded file name
        st.success(f"Uploaded: {pdf_file.name}")
        
    st.write("Results from LLM processing:")

    st.dataframe(pd.DataFrame(zip(COMPONENT_NAMES, [0.5,0,0.5,0,0]),columns=["component","%"]).set_index("component").transpose())
    if st.button("Add New Blend to Database"):
        # Here you would add the logic to save the new blend to the database
        st.success("New blend added to the database successfully!")




report_page()