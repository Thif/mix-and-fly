
import streamlit as st

with st.spinner("Loading app , please wait ...", show_time=True):
    from utils import BlendDataset,COMPONENT_NAMES,PROPERTY_NAMES,comp_dict
    from page_utils import remove_top




if "role" not in st.session_state:
    st.session_state.role = None

ROLES = ["Fuel Engineer","Process Engineer","Data Scientist"]


def login():
    remove_top()
    st.set_page_config(
        page_title="Login",
        #page_icon="📚",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    col1,col2=st.columns(2)

    with col1:
        st.image("./images/logo_desc.png",use_container_width=True)

    with col2:
        st.header("Log in")
        role = st.selectbox("Choose your role", ROLES)

        if st.button("Log in"):
            st.session_state.role = role
            st.rerun()


def logout():
    st.session_state.role = None
    st.rerun()


role = st.session_state.role

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

ds_readme_page = st.Page(
    "ds/read_me.py",
    title="Info",
    icon=":material/home:",
    default=(role == "Data Scientist"),
)

model_page = st.Page("ds/model.py", title="Model", icon=":material/modeling:")
explain_page = st.Page("ds/explainability.py", title="Explainability", icon=":material/graph_1:")
report_page = st.Page("ds/report.py", title="Report", icon=":material/report:")

fe_readme_page = st.Page(
    "fe/read_me.py",
    title="Info",
    icon=":material/home:",
    default=(role == "Fuel Engineer"),
)
monitoring_page = st.Page("pe/monitoring.py", title="Monitoring", icon=":material/monitor:")
document_page = st.Page("pe/document.py", title="Report", icon=":material/bug_report:")
summary_page = st.Page("pe/summary.py", title="Summary", icon=":material/folder:")

pe_readme_page = st.Page(
    "pe/read_me.py",
    title="Info",
    icon=":material/home:",
    default=(role == "Process Engineer"),
)
library_page = st.Page("fe/library.py", title="Library", icon=":material/database:")
design_page = st.Page("fe/design.py", title="Design", icon=":material/experiment:")
simulation_page = st.Page("fe/simulation.py", title="Simulation", icon=":material/candlestick_chart:")

account_pages = [logout_page]
ds_pages=[ds_readme_page,model_page,explain_page,report_page]
pe_pages=[pe_readme_page,monitoring_page,summary_page,document_page]
fe_pages=[fe_readme_page,library_page,simulation_page,design_page]




page_dict = {"Account": account_pages}
if st.session_state.role == "Data Scientist":
    page_dict["DS"] = ds_pages
if st.session_state.role == "Fuel Engineer":
    page_dict["FE"] = fe_pages
if st.session_state.role == "Process Engineer":
    page_dict["PE"] = pe_pages

if len(page_dict) > 1:
    pg = st.navigation( page_dict )
else:
    pg = st.navigation([st.Page(login)])

pg.run()