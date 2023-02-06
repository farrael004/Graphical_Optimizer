import json
import os
import sys
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from warnings import warn
from streamlit_lottie import st_lottie
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title='Hyperparameter results',
                   page_icon=":bar_chart:",
                   layout='wide')

tempPath = os.path.join(os.path.dirname(__file__), 'temp')
opt_id = sys.argv[-1]

def retrieve_experiments(df):
    """Function that writes to disk experiment results."""
    
    try:
        for filename in os.listdir(tempPath):
            tempfile = os.path.join(tempPath, filename)
            if filename[:10] != opt_id: continue  # check if file originates from this optimization session

            with open(tempfile, 'r') as openfile:
                try:
                    json_object = json.load(openfile)
                except:
                    warn("An error occurred when trying to read one of the experiment results.")
                else:
                    results = pd.DataFrame(json_object, index=[0])
                    if df.empty: df = results
                    else: df = pd.concat([df, results], ignore_index=True, axis=0)
            try:
                os.remove(tempfile)
            except:
                warn(f'Could not remove the temporary file {filename} from {tempPath}')

    except FileNotFoundError:
        raise FileNotFoundError(f"temp folder at {tempPath} not found.")
    
    finally:
        return df


def load_data():
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame()
    st.session_state['data'] = retrieve_experiments(st.session_state['data'])
    return st.session_state['data']


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True,
        filter=True, resizable=True, sortable=True
    )

    options.configure_side_bar()

    options.configure_selection("multiple", use_checkbox=False)
    selection = AgGrid(
        df,
        height=500,
        width='100%',
        fit_columns_on_grid_load=True,
        gridOptions=options.build()
    )

    return selection


@st.cache
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


dataset = st.container()
plotting = st.container()

df = load_data()
# lottie_image1 = load_lottie_url('https://assets1.lottiefiles.com/packages/lf20_VMNbpVQjTb.json')
# lottie_image2 = load_lottie_url('https://assets5.lottiefiles.com/packages/lf20_yMpiqXia1k.json')
# lottie_image3 = load_lottie_url('https://assets8.lottiefiles.com/packages/lf20_BgywoUBeiL.json')


with dataset:
    # st_lottie(lottie_image1, height=200)
    st.markdown('# Hyperparameter results 📅')
    grid_response = aggrid_interactive_table(df)

with plotting:
    st.markdown('# Plotter 📈')

    columns = []
    if grid_response["selected_rows"]:
        columns = list(grid_response["selected_rows"][0].keys())
        columns.remove('_selectedRowNodeInfo')

    col1, col2, col3 = st.columns(3)
    df1 = pd.DataFrame(grid_response["selected_rows"])
    y_axis = col1.multiselect('Y axis', columns)

    not_multicell_plotting = True
    if y_axis:
        plotting_data = pd.DataFrame()
        data_to_plot = []

        # Adding support for cells that contains a list of values
        for column in y_axis:
            for item in df1[column]:
                try:
                    new_list = list(map(float, item.replace('[', '').replace(']', '').split(',')))
                    i = df.loc[df[column] == item].index[0]
                    plot_name = column + f' {i}'
                    plotting_data[plot_name] = new_list
                    data_to_plot.append(plot_name)
                    not_multicell_plotting = False
                except:
                    plotting_data[column] = df1[column]
                    data_to_plot.append(column)
                    break

    if not_multicell_plotting:
        x_axis = col2.selectbox('X axis', columns)
    else:
        x_axis = col2.selectbox('X axis', ['index'])

    plot_type = col3.selectbox('Plot type', ['Line', 'Scatter'])

    if y_axis:
        
        plotting_data[x_axis] = df[column]
        
        if x_axis != 'index':
            plotting_data = plotting_data.sort_values(x_axis)
        
        if plot_type == 'Line':
            fig = px.line(plotting_data,
                          x=(x_axis if x_axis != 'index' else plotting_data.index),
                          y=data_to_plot)

        if plot_type == 'Scatter':
            fig = px.scatter(plotting_data,
                             x=(x_axis if x_axis != 'index' else plotting_data.index),
                             y=data_to_plot)

        fig.update_xaxes(gridcolor='black', showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(gridcolor='black', showline=True, linewidth=2, linecolor='black')
        # fig.update_layout(height=700)

        st.plotly_chart(fig)
    else:
        pass
        #lottie_image3 = load_lottie_file('123752-start-up-meeting.json')
        #st_lottie(lottie_image3, height=400)

hide_st_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
