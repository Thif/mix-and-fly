import streamlit as st
import pandas as pd
import plotly.express as px
from utils import COMPONENT_NAMES,PROPERTY_NAMES,comp_dict
from page_utils import remove_top
import pandas as pd
from datetime import datetime

def document_page():
    remove_top()
    st.set_page_config(
        page_title="Document",
        #page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Create a form for incident reporting
    with st.form(key='incident_form'):
        st.write("Please fill in the details of the incident:")

        # Input fields for the form
        incident_date = st.date_input("Incident Date", datetime.today())
        incident_time = st.time_input("Incident Time", datetime.now().time())
        incident_type = st.selectbox("Type of Incident", ["Wrong Fractions", "Deviation in Properties", "Other"])
        description = st.text_area("Description of the Incident", height=150)
        corrective_actions = st.text_area("Corrective Actions Taken", height=150)
        
        # Submit button
        submit_button = st.form_submit_button("Submit Incident Report")

        if submit_button:
            # Here you can add logic to save the incident report to a database or a file
            # For demonstration, we'll just display the submitted information
            st.success("Incident Report Submitted Successfully!")
            st.write("### Report Details:")
            st.write(f"**Incident Date:** {incident_date}")
            st.write(f"**Incident Time:** {incident_time}")
            st.write(f"**Type of Incident:** {incident_type}")
            st.write(f"**Description:** {description}")
            st.write(f"**Corrective Actions:** {corrective_actions}")

            # Optionally, you can save the report to a CSV file or a database
            # For example:
            # report_data = {
            #     "Date": incident_date,
            #     "Time": incident_time,
            #     "Type": incident_type,
            #     "Description": description,
            #     "Corrective Actions": corrective_actions
            # }
            # df = pd.DataFrame([report_data])
            # df.to_csv('incident_reports.csv', mode='a', header=False, index=False)






document_page()