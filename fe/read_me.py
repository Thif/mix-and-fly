import streamlit as st
from page_utils import remove_top

remove_top()
st.set_page_config(
    page_title="Info",
#        page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.header(f"{st.session_state.role} info page")


st.write(f"You will find here a list of the pages you have access to, each page contains a tool that will help you contribute to better design of sustainable aviation fuels ✈️ ")

with st.expander("⚙️ Library"):

    st.write(" In that page you will be able to get an overview of the blend database, both in terms of raw data and calculated metrics that are relevant for sustainable aviation fuels")
    st.markdown("- Filter volume fractions")
    st.markdown("- Ternary plot is updated with the filtered rows")
    st.markdown("- Select a blend within the database by clicking on the ternary plot")
    st.markdown("- Volume fractions and properties for the selection are displayed")
    st.page_link("fe/library.py", label="Go to page", icon="🚀")

with st.expander("⚙️ Simulation"):

    st.write("This section will help you simulate a fuel blend properties thanks to the machine learning model, it can be helpul to use that tool to get an estimate of a blend property before sending it to the lab")
    st.markdown("- Select the blend ID that will be used as a basis for the simulation")
    st.markdown("- The initial volume fractions of this blend ID are displayed")
    st.markdown("- The simulation starts with multiple volume fractions for this blend ID")
    st.markdown("- You can view the best metrics achieved and the ssociated volume fractions  ")
    st.markdown("- You also get an idea of the range of each properties during the simulation")
    st.page_link("fe/simulation.py", label="Go to page", icon="🚀")

with st.expander("⚙️ Design"):
    st.write("This tool complements the simulation, this time by providing the volume fractions needed to achieve a specific set of blend properties")
    st.markdown("- Select the blend property values you want to achieve")
    st.markdown("- The model will find the corresponding volume fractions and display the calculated metrics for that blend")
    st.page_link("fe/design.py", label="Go to page", icon="🚀")