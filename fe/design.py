import streamlit as st
import plotly.graph_objects as go
from utils import BlendDataset, COMPONENT_NAMES, PROPERTY_NAMES,PREDICTED_PROPERTY_NAMES
import numpy as np
from page_utils import remove_top

def design_page():

    remove_top()
    st.set_page_config(
        page_title="Blend Design",
        layout="wide",
        initial_sidebar_state="collapsed"
    )



    col1, col2 = st.columns([2,3])


    with col1:
        st.write("Select a set of target properties:")
        col11,col12=st.columns(2)
        
        with col11:
            fractions = []

            for p in PROPERTY_NAMES[:5]:
                # Create a range slider for each component
                fraction_range = st.slider(
                    f"{p}",
                    -4.0, 4.0, 0.0 # Step size for the slider
                )
                fractions.append(fraction_range)

        with col12:
            for p in PROPERTY_NAMES[5:]:
                # Create a range slider for each component
                fraction_range = st.slider(
                    f"{p}",
                    -4.0, 4.0, 0.0 # Step size for the slider
                )
                fractions.append(fraction_range)

            BD = BlendDataset()
            df = BD.load_data()
            df=BD.rename_cols(df)
            df=BD.add_metrics(df)


            # Calculate the Euclidean distance from the given row to each row in the DataFrame
            distances = np.linalg.norm(df[PROPERTY_NAMES].values - fractions, axis=1)

            # Find the index of the closest row
            closest_index = np.argmin(distances)

            # Get the closest row
            closest_row = df.iloc[closest_index]
            

            


    with col2:
        st.subheader("Calculated metrics and volume fractions")
        col21,col22,col23,col24=st.columns(4)
        with col21:
            st.metric(label="Performance", value=f'{round(closest_row["Performance"] * 100):.0f} %',border=True)
        with col22:
            st.metric(label="Safety", value=f'{round(closest_row["Safety"] * 100):.0f} %',border=True)
        with col23:
            st.metric(label="Sustainability", value=f'{round(closest_row["Sustainability"] * 100):.0f} %',border=True)
        with col24:
            st.metric(label="Cost", value=f'{round(closest_row["Cost"] ,2):.2f} $/l',border=True)
            # Property names



        # Create a radar plot
        fig = go.Figure()

        r_values = closest_row[COMPONENT_NAMES].values.tolist() + [closest_row[COMPONENT_NAMES].values[0]]  # Close the loop
        theta_values = COMPONENT_NAMES + [COMPONENT_NAMES[0]]  # Close the loop

        # Add a trace for the radar plot
        fig.add_trace(go.Scatterpolar(
            r=r_values,  # Close the loop
            theta=theta_values,  # Close the loop
            fill='toself',
            name='Selected Properties',
            
        ))


        # Update traces to set the box color
        fig.update_traces(marker_color="teal", line_color='teal')  # Set line color to red


        # Update layout for better visualization
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(r_values)]  # Set the range based on your data
                )
            ),
            showlegend=False,
            plot_bgcolor="#0E1117"
        )

        # Display the radar plot in Streamlit
        st.plotly_chart(fig)

design_page()