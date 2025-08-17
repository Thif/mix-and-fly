import streamlit as st
from page_utils import remove_top
from ds.model import model_page

remove_top()
st.set_page_config(
    page_title="Info",
#        page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.header(f"{st.session_state.role} info page")


st.write(f"You will find here a list of the pages you have access to, each page contains a tool that will help you contribute to better design of sustainable aviation fuels ✈️ ")

with st.expander("➕ Model"):

    st.write("In this section you can check the cross validation metrics on the models, and investigate potential outliers")
    st.markdown("- Select the blend property of interest")
    st.markdown("- Validation data for selected models is displayed")

    st.page_link("ds/model.py", label="Go to page", icon="🚀")

with st.expander("➕ Explainability"):

    st.write("In this section you can explore the contribution of the features to each blend property model")
    st.markdown("- Select the blend property of interest")
    st.markdown("- Feature contribution (SHAP values) for selected models is displayed")

    st.page_link("ds/explainability.py", label="Go to page", icon="🚀")

with st.expander("➕ Add blend"):

    st.write("In this section you can add a new blend to the database")
    st.markdown("- Add the pdf report from the Certificate")
    st.markdown("- The report is processed by the LLM and the data points are displayed")
    st.markdown("- Add the new blend to the database")

    st.page_link("ds/report.py", label="Go to page", icon="🚀")
    