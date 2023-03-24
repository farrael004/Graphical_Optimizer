import sys
import os
import json
import numpy as np
from warnings import warn
import pandas as pd
import streamlit as st
import altair as alt

from st_aggrid import GridOptionsBuilder, AgGrid

st.set_page_config(layout="wide")

tempPath = os.path.join(os.path.dirname(__file__), 'temp')
opt_id = sys.argv[-2]
api_url = sys.argv[-1]


def old_method(df):
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
                    if df.empty:
                        df = results
                    else:
                        df = pd.concat([df, results], ignore_index=True, axis=0)
            try:
                os.remove(tempfile)
            except:
                warn(f'Could not remove the temporary file {filename} from {tempPath}')

    except FileNotFoundError:
        raise FileNotFoundError(f"temp folder at {tempPath} not found.")

    finally:
        return df


def retrieve_experiments(df):
    """Function that writes to disk experiment results."""

    return old_method(df)


def load_data():
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame()
    st.session_state['data'] = retrieve_experiments(st.session_state['data'])
    return st.session_state['data']


df = load_data()
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

    options = st.multiselect(
        'Which columns do you want in the plot?',
        df.select_dtypes(include=np.number).columns.tolist(),
        ['Adjusted R^2 Score'],
        max_selections=3
    )

    multi_value = st.checkbox('Display the Test Loss parameter values over the runs?')
    df1 = df.copy(deep=True)
    df1 = df1[['Train loss', 'Test loss']]

    df1_total = pd.DataFrame()

    for i in range(df1.shape[0]):

        df1_test_expand = df1.loc[i, ['Test loss']].apply(pd.Series)

        df1_test_transpose = df1_test_expand.T
        df1_test_transpose.rename(columns={'Test loss': 'Score'}, inplace=True)
        df1_test_transpose['Iteration'] = i
        df1_test_transpose['Run Number'] = df1_test_transpose.index

        df1_total = pd.concat([df1_total, df1_test_transpose], ignore_index=True)

    # st.write(df1_total)
    df1_display = pd.DataFrame(columns=['Score', 'Iteration', 'Run Number'])

    if not selected_df.empty:
        indices = list(selected_df['index_column'])
        df1_display = df1_total[df1_total['Iteration'].isin(indices)]

    if multi_value:

        c1 = alt.Chart(df1_display).mark_line().encode(
            x='Run Number:Q',
            y='Score:Q',
            color='Iteration:N'
        )
        st.altair_chart(c1, use_container_width=True)

    options.append('index_column')
    # st.write(options)

    chart_data = pd.DataFrame(columns=options)
    if not selected_df.empty:
        chart_data = selected_df.loc[:, options]

    chart_data = pd.melt(chart_data, id_vars=['index_column'], var_name="metric", value_name="score")

    chart = alt.Chart(data=chart_data, mark=cht_mark).encode(
        x='metric',
        y='score',
        tooltip=['metric', 'score', 'index_column'],
        color=alt.Color('metric:N'),
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
