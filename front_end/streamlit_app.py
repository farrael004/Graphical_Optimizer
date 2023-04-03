import sys
import os
import json
import ast
import requests
from warnings import warn
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

from st_aggrid import GridOptionsBuilder, AgGrid

st.set_page_config(layout="wide")
    
tempPath = os.path.join(os.path.dirname(__file__), 'temp')
api_url = sys.argv[-1]

def old_method(df):
    try:
        for filename in os.listdir(tempPath):
            tempfile = os.path.join(tempPath, filename)
            #if filename[:10] != opt_id: continue  # check if file originates from this optimization session

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

def retrieve_experiments(id):
    try:
        params = {'id': id}
        res = requests.get(api_url + '/api/data', params=params)
        res.raise_for_status()
    except Exception as e:
        st.error(e)
        st.stop()
    
    return pd.read_json(res.content.decode("utf-8"))

def retrieve_experiments_names():
    try:
        res = requests.get(api_url + '/api/experiments')
        res.raise_for_status()
        
    except Exception as e:
        st.error(e)
        st.stop()
    
    return ast.literal_eval(res.content.decode("utf-8"))

def load_data(id):
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame()
    st.session_state['data'] = retrieve_experiments(id)
    return st.session_state['data']

id = st.selectbox('Experiments', retrieve_experiments_names())

df = load_data(id)
if df.empty: st.stop()

col1, col2 = st.columns(2)
cht_mark = 1

with col1:
    st.header('Results of the experiment: ')

    chart_types = ['circle', 'point', 'square', 'tick']
    cht_mark = st.selectbox('Which type of chart do you want?', chart_types, 1)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True, groupSelectsChildren=True,
                           groupSelectsFiltered=True, pre_selected_rows=[0])
    gb.configure_column(df.columns[0], headerCheckboxSelection=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()

    df['index_column'] = df.index

    grid_response = AgGrid(df, gridOptions=gridOptions)

    selected = grid_response['selected_rows']
    # st.write(grid_response['selected_rows'])
    selected_df = pd.DataFrame(selected)

with col2:
    st.header('Plot from the experiment: ')
    # st.write(df.columns.tolist())

    l1 = df.select_dtypes(include=object).columns.tolist()
    l1 = [x for x in l1 if type(df.loc[0, x]) is list]

    list_options = st.multiselect(
        'Which list type columns do you want in the plot?',
        options=l1,
        default=None,
        max_selections=1
    )

    df1 = df.copy(deep=True)
    df1_total = pd.DataFrame()

    if list_options:
        df1 = df1[list_options]

        for i in range(df1.shape[0]):
            df1_test_expand = df1.loc[i, list_options].apply(pd.Series)

            df1_test_transpose = df1_test_expand.T
            df1_test_transpose.rename(columns={list_options[0]: 'Score'}, inplace=True)
            df1_test_transpose['Iteration'] = i
            df1_test_transpose['Run Number'] = df1_test_transpose.index

            df1_total = pd.concat([df1_total, df1_test_transpose], ignore_index=True)

        df1_display = pd.DataFrame(columns=['Score', 'Iteration', 'Run Number'])

        if not selected_df.empty:
            indices = list(selected_df['index_column'])
            df1_display = df1_total[df1_total['Iteration'].isin(indices)]

        if df1_display['Score'].dtype == object:
            c1 = alt.Chart(df1_display).mark_point().encode(
                x='Run Number:Q',
                y='Score:N',
                color='Iteration:N'
            ).interactive()
            st.altair_chart(c1, use_container_width=True)
        else:
            c1 = alt.Chart(df1_display).mark_line().encode(
                x='Run Number:Q',
                y='Score:Q',
                color='Iteration:N'
            ).interactive()
        st.altair_chart(c1, use_container_width=True)

col1, col2, col3 = st.columns(3)
y_axis = col1.selectbox(
    'Y axis',
    df.select_dtypes(include=np.number).columns.tolist()
)

x_axis_list = [item for item in df.select_dtypes(include=np.number).columns.tolist() if item != y_axis]

x_axis = col2.selectbox(
    'X axis',
    x_axis_list
)

color_list = [item for item in x_axis_list if item != x_axis]

color = col3.selectbox(
    'Color',
    color_list
)
if color is None:
        color = 'metric:N'

if y_axis and x_axis:
    options = [y_axis, x_axis, color]

    chart_data = pd.DataFrame(columns=options)
    if not selected_df.empty:
        chart_data = selected_df.loc[:, options]

    chart = alt.Chart(chart_data).mark_point().encode(
        x=alt.X(x_axis, type='quantitative'),
        y=alt.Y(y_axis, type='quantitative'),
        tooltip=['metric', 'score', x_axis],
        color=alt.Color('metric:N'),
    ).interactive()
    
    chart = alt.Chart(chart_data).mark_circle().encode(
        x=alt.X(x_axis, type='quantitative'),
        y=alt.Y(y_axis, type='quantitative'),
        tooltip=options,
        color=alt.Color(color),
    ).interactive()

    st.altair_chart(chart, use_container_width=True)