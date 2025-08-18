import streamlit as st
from page_utils import remove_top

remove_top()
st.set_page_config(
    page_title="Info",
#        page_icon="📚",
    layout="wide",
    #initial_sidebar_state="collapsed"
)

st.header(f"{st.session_state.role} info page")


st.write(f"You will find here a list of the pages you have access to, each page contains a tool that will help you contribute to better design of sustainable aviation fuels ✈️ ")

with st.expander("➕ Monitoring"):

    st.write(" In that page you will be able to monitor the volume fractions and expected properties from the SAF process in real time")
    st.markdown("- Specify the target volume fractions for the process")
    st.markdown("- Blend properties are simulated for the new fractions and used as target")
    st.markdown("- You can start the streaming !")
    st.markdown("- You visualize the values of the fractions and properties, together with their targets")
    st.markdown("- In case of deviation, the faulty parameter is highlighted and the incident recorded")
    st.page_link("pe/monitoring.py", label="Go to page", icon="🚀")

with st.expander("➕ Deviation summary"):
    st.markdown("- Check the deviations that occured, when and which parameter")
    st.page_link("pe/summary.py", label="Go to page", icon="🚀")

with st.expander("➕ Document deviation"):
    st.write(" In that page you will be able to document deviations and non compliance of the blend during the production process")
    st.markdown("- Fill in the form with the date and the incident")
    st.markdown("- Send the form to the relevant stakeholders")
    st.page_link("pe/document.py", label="Go to page", icon="🚀")



# Example session state for favorites
if "deviations" in st.session_state:
    num_deviations = len(st.session_state.deviations) 
    st.markdown(f"##### ⚠️ You have **{num_deviations}** deviation(s) flagged in the process")
else:
    st.markdown(f"##### ⚠️ No process deviation yet, time to check the monitoring ?")
