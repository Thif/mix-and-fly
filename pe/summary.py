import streamlit as st
import pandas as pd
import plotly.express as px
from utils import COMPONENT_NAMES,PROPERTY_NAMES,comp_dict
from page_utils import remove_top
import pandas as pd
from datetime import datetime

def summary_page():
    remove_top()
    st.set_page_config(
        page_title="Summary",
        #page_icon="📚",
        layout="wide",
        #initial_sidebar_state="collapsed"
    )
    st.markdown(
        """
        <h1 style="text-align: center;"> Deviations Summary</h1>
        """,
        unsafe_allow_html=True
    )
    if "deviations" in st.session_state:

        st.dataframe(pd.DataFrame(st.session_state.deviations),hide_index=True)
    else:
        st.write("No deviations recorded !")

summary_page()