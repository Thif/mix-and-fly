import streamlit as st
import plotly.express as px
import pandas as pd
from utils import BlendDataset,COMPONENT_NAMES,PROPERTY_NAMES
from page_utils import remove_top


@st.cache_data
def load_df_bd():
    BD=BlendDataset()
    df=BD.load_data()
    df=BD.rename_cols(df)
    df=BD.add_metrics(df)
    return df

def library_page():

    remove_top()

    st.set_page_config(
        page_title="Blend Library",
#        page_icon="📚",
        layout="wide",
       # initial_sidebar_state="collapsed"
    )

    selected_index=None
    col1, col2,col3 = st.columns([1, 3,2])

    with col1:
        st.write("Volume fraction filters")
        fractions = []

        for i in range(5):
            # Create a range slider for each component
            fraction_range = st.slider(
                f"{COMPONENT_NAMES[i]}",
                0.0, 0.5, (0.0, 0.5),  # Set the default range (min, max)
                step=0.01  # Step size for the slider
            )
            fractions.append(fraction_range)

        is_compliant = st.checkbox("Jet-A compliant")
            
    with col2:

        df=load_df_bd()

  
        df = df[
        (df['Diesel Fraction'].between(fractions[0][0], fractions[0][1])) &
        (df['Biofuel Fraction'].between(fractions[1][0], fractions[1][1])) &
        (df['Synthetic Fuel Fraction'].between(fractions[2][0], fractions[2][1])) &
        (df['Additives Fraction'].between(fractions[3][0], fractions[3][1])) &
        (df['Waste Oil Fraction'].between(fractions[4][0], fractions[4][1]))
    ]
        df=df[df.Compliant==is_compliant]

        # Create a ternary scatter plot
        fig = px.scatter_ternary(df, a="Safety", b="Performance", c="Sustainability", 
                                    #color="Compliant",
                                    size="Cost",
                                    opacity=0.8,
                                  labels={"Safety": "Safety", "Performance": "Performance", "Sustainability": "Sustainability"},
                                  hover_name="ID",
                                          width=500,  
        height=500 ,
        #color_continuous_scale='GnBu'
        )
        fig.update_traces(marker=dict(color="teal",size=10))



        # Display the plot in Streamlit
        selected_data = st.plotly_chart(fig, on_select="rerun")
        if len(selected_data.selection.point_indices)>0:
            selected_index=selected_data.selection.point_indices[0] # Access selected point indices


    with col3:
        if selected_index:
            df["plot_index"]=df.reset_index().index
            filtered_df = df[df.plot_index==selected_index]

            #fractions
            st.write("selected blend details:")
            df_frac=filtered_df[COMPONENT_NAMES].transpose()

            st.dataframe(df_frac)

            if filtered_df["Compliant"].values[0]:
                st.success("✅ Jet-A compliant")
            else:
                st.error("❌ Not Jet-A compliant")

            # Properties
            transposed_df=filtered_df[PROPERTY_NAMES].melt()


            # Create a vertical bar chart for the transposed DataFrame
            fig = px.bar(transposed_df, x="variable",
             y="value", 
             labels={'x': 'Properties', 'y': 'Scores'},
             height=300,
             )

            fig.update_traces(texttemplate='%{value:.2f}', textposition='inside',marker_color='teal', marker_line_color='teal', marker_line_width=1)
            # Display the bar chart in Streamlit
            fig.update_layout(
                xaxis_title=None,  # Hide x-axis title
                yaxis_title=None,  # Hide y-axis title
                showlegend=False    # Optionally hide the legend if not needed
            )
            st.plotly_chart(fig)
        else:
            st.subheader("Please click on a point in the ternary plot to see details 👀")


library_page()
