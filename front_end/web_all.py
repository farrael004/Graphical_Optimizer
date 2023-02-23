import sys
import os
import json
import numpy as np
import requests
from warnings import warn
import pandas as pd
import streamlit as st
import altair as alt

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

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

    # try:
    #    res = requests.get(api_url + '/api/data')
    #    res.raise_for_status()
    # except Exception as e:
    #    st.error(e)
    #    st.stop()

    # return pd.read_json(res.content.decode("utf-8"))


def load_data():
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame()
    st.session_state['data'] = retrieve_experiments(st.session_state['data'])
    return st.session_state['data']


df = load_data()
if df.empty: st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header('Results of the experiment: ')
    # st.dataframe(df)
    # performance_parameter = st.selectbox('Performance parameter', df.columns)
    # min_or_max = st.radio('Minimize or Maximize', ['Minimize', 'Maximize'])
    # if min_or_max == 'Maximize':
    #     best_score = df[performance_parameter].max()
    # else:
    #     best_score = df[performance_parameter].min()
    # st.caption('Best performance is: ')
    # st.write(best_score)
    # st.caption('Combination of hyperparameters for this is: ')
    # st.write(opt.results.best_params_)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True, groupSelectsChildren=True,
                           groupSelectsFiltered=True, pre_selected_rows=[0])
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()

    grid_response = AgGrid(df, gridOptions=gridOptions)

    selected = grid_response['selected_rows']
    # st.write(grid_response['selected_rows'])
    selected_df = pd.DataFrame(selected)

    df['index_column'] = df.index

with col2:
    st.header('Plot from the experiment: ')
    # st.write(df.columns.tolist())

    options = st.multiselect(
        'Which columns do you want in the plot?',
        df.select_dtypes(include=np.number).columns.tolist(),
        ['Adjusted R^2 Score']
    )

    options.append('index_column')
    # st.write(options)

    chart_data = df.loc[:, options]
    if not selected_df.empty:
        chart_data = selected_df.loc[:, options]

    # chart_data = df.loc[:, options].assign(source=0)
    # if not selected_df.empty:
    #     selected_data = selected_df.loc[:, options].assign(source='selection')
    #     chart_data = pd.concat([chart_data, selected_data])

    # chart_data = pd.melt(chart_data, id_vars=options[0], var_name="metric", value_name="score")
    # chart_data = pd.melt(chart_data, id_vars=['Adjusted R^2 Score'], var_name="metric", value_name="score")
    # st.dataframe(chart_data)

    chart_data = pd.melt(chart_data, id_vars=['index_column'], var_name="metric", value_name="score")

    # chart = alt.Chart(data=chart_data).mark_bar().encode(
    #     x=alt.X("metric:O"),
    #     y=alt.Y("mean(score):Q", stack=False),
    #     # color=alt.Color('source:N', scale=alt.Scale(domain=['selection'])),
    # )
    #
    # st.altair_chart(chart, use_container_width=True)

    chart = alt.Chart(data=chart_data).mark_point().encode(
        x='metric',
        y='score',
        tooltip = ['metric', 'score', 'index_column'],
        color=alt.Color('metric:N'),
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # with st.form('my_form'):
    #     options = st.multiselect(
    #         'Which columns do you want in the plot?',
    #         df.columns.to_list(),
    #         ['Adjusted R^2 Score']
    #     )
    #     submitted = st.form_submit_button('Submit')
    #     if submitted:
    #         st.line_chart(df, y=options)
