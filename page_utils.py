import streamlit as st

def remove_top():
    st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.logo("images/logo_page.png")

