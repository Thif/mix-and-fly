import streamlit as st
import plotly.express as px
from utils import BlendDataset, COMPONENT_NAMES, PROPERTY_NAMES,PREDICTED_PROPERTY_NAMES
from page_utils import remove_top

INIT_ID=456

def simulation_page():

    remove_top()

    st.set_page_config(
        page_title="Blend Simulator",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


    col1, col2 = st.columns([1, 4])

    with col1:
        BD = BlendDataset()
        df = BD.load_data()
        df=BD.rename_cols(df)


        # Create a slider to select the row ID
        selected_id = st.slider("Select a blend ID to simulate:", min_value=df.ID.min(), max_value=df.ID.max(), value=INIT_ID)

        # Filter the DataFrame based on the selected ID
        selected_row = df[df['ID'] == selected_id]

        # Display the selected row
        st.write("Initial fractions:")
        st.dataframe(selected_row[COMPONENT_NAMES].transpose())

        if "selected_id" not in st.session_state:
            st.session_state.selected_id = None
            st.session_state.metric_df=None


        if selected_id != st.session_state.selected_id:
            #get initial metric values

            df_init=BD.add_metrics(df)
            st.session_state.init_perf=df_init["Performance"].values[0]
            st.session_state.init_safe=df_init["Safety"].values[0]
            st.session_state.init_sus=df_init["Sustainability"].values[0]
            st.session_state.init_cost=df_init["Cost"].values[0]

            #generate random fractions
            df = BD.generate_samples_from_id(df, selected_id)
            df = BD.add_properties(df)
            df=BD.rename_cols(df)
            df=BD.add_metrics(df, predicted=True)
            st.session_state.selected_id = selected_id  # Store the current selected ID
            st.session_state.metric_df = df  # Store the current selected ID
            

    with col2:

        df=st.session_state.metric_df
        if st.session_state.metric_df is not None:

            st.write("Best metrics with corresponding volume fractions")
            col_metric, col_frac = st.columns([2, 1])
            
            
            with col_metric:
                
                options = ["Sustainability", "Safety", "Performance", "Cost"]
                selection = st.pills("",options, default="Performance")


                df_selected_max = df[df[selection] == df[selection].max()]
                df_selected_min = df[df[selection] == df[selection].min()]
                new_value_min=round(df_selected_min[selection].values[0], 2)
                new_value_max=round(df_selected_max[selection].values[0], 2)


                if selection=="Cost":
                    st.metric(label=selection, value=f'{round(new_value_min,2)} $/l', delta=round(new_value_min-st.session_state.init_cost,2), border=True,delta_color="inverse")
                elif selection=="Performance":
                    st.metric(label=selection, value=f'{round(new_value_max*100)} %', delta=round(new_value_max-st.session_state.init_perf,2), border=True)
                elif selection=="Sustainability":
                    st.metric(label=selection, value=f'{round(new_value_max*100)} %', delta=round(new_value_max-st.session_state.init_sus,2), border=True)
                elif selection=="Safety":
                    st.metric(label=selection, value=f'{round(new_value_max*100)} %', delta=round(new_value_max-st.session_state.init_safe,2), border=True)
            with col_frac:
                if selection=="Cost":
                    data_min=df_selected_min[COMPONENT_NAMES+["Compliant"]].iloc[0]
                    st.dataframe(data_min.iloc[:5].transpose())
                    if data_min["Compliant"]:
                        st.success("✅ Jet-A compliant")
                    else:
                        st.error("❌ Not Jet-A compliant")
                else:
                    data_max=df_selected_max[COMPONENT_NAMES+["Compliant"]].iloc[0]
                    st.dataframe(data_max.iloc[:5].transpose())
                    if data_max["Compliant"]:
                        st.success("✅ Jet-A compliant")
                    else:
                        st.error("❌ Not Jet-A compliant")

            
            df_melted = df[PREDICTED_PROPERTY_NAMES].melt()

            # Create a box plot using Plotly Express
            fig = px.box(df_melted, x='variable', y='value', title="Simulated properties range")



            # Update traces to set the box color
            fig.update_traces(marker_color="#1ABC9C", line_color="#1ABC9C")  # Set line color to red


            # Display the box plot in Streamlit
            st.plotly_chart(fig)


simulation_page()