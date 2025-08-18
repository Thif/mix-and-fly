import streamlit as st
import pandas as pd
import time
import random
from utils import BlendDataset, COMPONENT_NAMES, PROPERTY_NAMES,PREDICTED_PROPERTY_NAMES,ORIGINAL_COMPONENT_NAMES
import numpy as np
from page_utils import remove_top
import plotly.graph_objects as go
from datetime import datetime

INIT_ID=546
max_steps=30
PARAMS=COMPONENT_NAMES+["Density","Particulate Matter"]
THRESHOLD=0.06

@st.cache_data
def get_selected_row(selected_id,targets):
    #get properties
    BD = BlendDataset()
    df = BD.load_data()
    # Filter the DataFrame based on the selected ID
    selected_row = df[df['ID'] == selected_id]
    selected_row.loc[:,ORIGINAL_COMPONENT_NAMES]=targets
    selected_row=BD.add_properties(selected_row)
    return selected_row

def monitoring_page():

    remove_top()
    st.set_page_config(
        page_title="Blend Real-Time Monitoring",
        layout="wide",
        #initial_sidebar_state="collapsed"
    )

    col1,col3=st.columns([1,4])

    with col1:
        st.write("Specify target fractions:")
        # Create a list to store the target values
        targets = []

        for i, c in enumerate(COMPONENT_NAMES):
            # Create a number input in the corresponding column
            target = st.number_input(
                f"{c}",
                min_value=0.0,  # Set a minimum value if needed
                max_value=1.0,  # Set a maximum value if needed
                value=0.2,      # Default value
                step=0.01       # Step size for the input
            )
            targets.append(target)

        # Calculate the sum of the list
        total = sum(targets)

        # Check if the sum is greater than 1
        if total != 1:
            st.warning(f"The sum of the values must equal 1! ( current total: {total:.2f}. Please check your inputs.")
            target_density,target_pm=-1,-1
        else:
            selected_id = st.slider("Blend ID used as a basis:", min_value=1, max_value=2000, value=INIT_ID)

            selected_row=get_selected_row(selected_id,targets)

            target_density,target_pm=selected_row[["Density (Predicted)","Particulate Matter (Predicted)"]].values[0]
        
        if st.button("▶️ Start Streaming"):
            st.session_state.streaming = True  # Set streaming state to True

        if st.button("⏹️ Stop Streaming"):
            st.session_state.streaming = False  # Set streaming state to False


    with col3:

        num_bars = 7  # Number of bars to create
        bar_values = targets + [target_density,target_pm]  # Initial values for the bars
        max_thresholds = targets + [target_density,target_pm]  # Define maximum thresholds for each bar
        timestamps=[]
        density_ts=[]
        pm_ts=[]
        t=0

        # Initialize streaming state if not already set
        if 'streaming' not in st.session_state or total!=1:
            st.session_state.streaming = False
            st.subheader("Click the 'Start streaming' button to start monitoring !")

        col31,col32=st.columns([2,1])

        with col31:
            pid_placeholder = st.empty()
            df_placeholder = st.empty()
            deviation_count_placeholder=st.empty()
            
        with col32:
            
            line_placeholder = st.empty()
            line2_placeholder = st.empty()

        # Simulation loop
        while True:
            if st.session_state.streaming:
                # Update data for each bar

                frequency = 0.01  # Frequency of the sine wave
                amplitude = 0.04  # Amplitude of the oscillation
                noise_level = 0.01  # Level of random noise

                if "deviations" not in st.session_state:
                    st.session_state.deviations = []

                for j in range(num_bars):
                    sine_wave = amplitude * np.sin(2 * np.pi  * frequency * (t-j*10))
                    bar_values[j] = max_thresholds[j] *(1+ sine_wave+np.random.normal(0, noise_level))  # Replace with your real data source
                    if abs((bar_values[j] - max_thresholds[j])/bar_values[j])>THRESHOLD:
                        st.session_state.deviations+=[{"date":datetime.now(),"parameter":PARAMS[j],"value":bar_values[j],"target":max_thresholds[j]}]




                
                #update colors
                shape_colors=["#0E1117" if abs((bar_values[i] - max_thresholds[i])/bar_values[i])<THRESHOLD else "teal" for i in range(num_bars)]
                
                #time values
                timestamps+=[t]
                density_ts+=[bar_values[5]]
                pm_ts+=[bar_values[6]]

                # Create a Plotly figure for the P&ID diagram
                fig_pid = go.Figure()

                # Set axis limits
                fig_pid.update_xaxes(range=[0, 10], visible=False)
                fig_pid.update_yaxes(range=[0, 10], visible=False)

                # Volume fraction measurement boxes (VF1 to VF5)
                vf_positions = [(0, 8), (0, 6.5), (0, 5), (0, 3.5), (0, 2)]
                for i, (x, y) in enumerate(vf_positions):
                    fig_pid.add_shape(type="rect", x0=x, y0=y, x1=x + 1.5, y1=y + 1,
                                      line=dict(color="white"), fillcolor=shape_colors[i])
                    fig_pid.add_annotation(x=x + 0.75, y=y + 0.5,
                                           text=f"{' '.join(COMPONENT_NAMES[i].split()[:-1])}", showarrow=False)

                    # Arrows to mixer
                    fig_pid.add_shape(type="line", x0=x + 1.5, y0=y + 0.5, x1=4, y1=6,
                                      line=dict(color="white", width=1))

                # Mixer box positioned between VF and Density
                fig_pid.add_shape(type="rect", x0=4, y0=5.5, x1=6, y1=7,
                                  line=dict(color="white"), fillcolor="#0E1117")
                fig_pid.add_annotation(x=5, y=6.25, text="Mixer", showarrow=False)

                # Arrow from mixer to density
                fig_pid.add_shape(type="line", x0=6, y0=6.25, x1=7.5, y1=7.5,
                                  line=dict(color="white", width=1))

                # Density measurement box
                fig_pid.add_shape(type="rect", x0=7.5, y0=7, x1=9, y1=8.5,
                                  line=dict(color="white"), fillcolor=shape_colors[5])
                fig_pid.add_annotation(x=8.25, y=7.75, text="Density", showarrow=False)

                # Arrow from mixer to particulate matter
                fig_pid.add_shape(type="line", x0=6, y0=5.75, x1=7.5, y1=4.5,
                                  line=dict(color="white", width=1))

                # Particulate Matter measurement box
                fig_pid.add_shape(type="rect", x0=7.5, y0=4, x1=9, y1=5.5,
                                  line=dict(color="white"), fillcolor=shape_colors[6])
                fig_pid.add_annotation(x=8.25, y=4.75, text="PM", showarrow=False)

                # General layout settings
                fig_pid.update_layout(
                   # title="P&ID Diagram - Volume Fractions to Mixer and Measurements",
                    showlegend=False,
                    #width=800,
                    height=500,  # Adjust height if needed
                    #plot_bgcolor="black"
                )

                # Display the P&ID diagram
                pid_placeholder.plotly_chart(fig_pid, use_container_width=True,key=f"pid_diagram{random.uniform(0,1)}")

                # Create a DataFrame
                data = {
                    'Value': bar_values[:5],
                }
                df = pd.DataFrame(data, index=COMPONENT_NAMES).T
                df.columns=[c.split()[0] for c in df.columns]

                # Function to highlight values above the threshold


                highlight_cols = [COMPONENT_NAMES[i].split()[0] for i in range(len(COMPONENT_NAMES)) if abs(bar_values[i] - max_thresholds[i]) > THRESHOLD]

                if len(highlight_cols)>0:
                # Display the DataFrame with highlighted values
                    df_placeholder.dataframe(df.style.set_properties(subset=highlight_cols, **{'background-color': "teal"}))
                else:
                    df_placeholder.dataframe(df)

                fig_line = go.Figure()
                # Add maximum threshold lines
                fig_line.add_trace(go.Scatter(
                    x=timestamps[-max_steps:],
                    y=density_ts[-max_steps:],
                    mode='lines',
                    name='Max Threshold',
                    line=dict(color='white', width=2),  # Dashed line for the threshold
                ))

                y_target=[target_density]*len(timestamps)
                # Add maximum threshold lines
                fig_line.add_trace(go.Scatter(
                    x=timestamps[-max_steps:],
                    y=y_target[-max_steps:],
                    mode='lines+text',
                    name='Max Threshold',
                    line=dict(color='teal', width=2, dash='dash'),  # Dashed line for the threshold
                    #text=f'Target: {target_density:.2f}',  # Text above the line
                    textposition='top center'  # Position the text above the line
                ))

                # Update layout for the bar chart
                fig_line.update_layout(
                    #title="Real-Time Bar Chart",
                    #xaxis_title="Components",
                    yaxis_title="Density",
                    height=300,  # Set the height of the plot
                    showlegend=False
                )
                # Update the chart in the placeholder
                line_placeholder.plotly_chart(fig_line, use_container_width=True, key=f"line_chart{random.uniform(0,1)}")

                fig_line2 = go.Figure()
                # Add maximum threshold lines
                fig_line2.add_trace(go.Scatter(
                    x=timestamps[-max_steps:],
                    y=pm_ts[-max_steps:],
                    mode='lines',
                    name='Max Threshold',
                    line=dict(color='white', width=2),  # Dashed line for the threshold
                ))

                y_target_pm=[target_pm]*len(timestamps)
                # Add maximum threshold lines
                fig_line2.add_trace(go.Scatter(
                    x=timestamps[-max_steps:],
                    y=y_target_pm[-max_steps:],
                    mode='lines+text',
                    name='Max Threshold',
                    line=dict(color='teal', width=2, dash='dash'),  # Dashed line for the threshold
                    #text=f'Target: {target_density:.2f}',  # Text above the line
                    textposition='top center'  # Position the text above the line
                ))

                # Update layout for the bar chart
                fig_line2.update_layout(
                    #title="Real-Time Bar Chart",
                    #xaxis_title="Components",
                    yaxis_title="Particulate Matter",
                    height=300,  # Set the height of the plot
                    showlegend=False
                )
                # Update the chart in the placeholder
                line2_placeholder.plotly_chart(fig_line2, use_container_width=True, key=f"line_chart{random.uniform(0,1)}")

                # Wait for a second before next update
                time.sleep(1)
                t+=1
                deviation_count_placeholder.write(f"Number of deviations recorded: {len(st.session_state.deviations)}")
            else:
                # If not streaming, break the loop
                break



monitoring_page()
