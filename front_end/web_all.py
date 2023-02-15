import sys
import os
import json
import requests
from warnings import warn
import pandas as pd
import streamlit as st

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

def retrieve_experiments(df):
    """Function that writes to disk experiment results."""
    
    
    return old_method(df)
    
    #try:
    #    res = requests.get(api_url + '/api/data')
    #    res.raise_for_status()
    #except Exception as e:
    #    st.error(e)
    #    st.stop()
    
    #return pd.read_json(res.content.decode("utf-8"))

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
    st.dataframe(df)
    performance_parameter = st.selectbox('Performance parameter', df.columns)
    min_or_max = st.radio('Minimize or Maximize',['Minimize', 'Maximize'])
    if min_or_max == 'Maximize': best_score = df[performance_parameter].max()
    else: best_score = df[performance_parameter].max()
    st.caption('Best performance is: ')
    st.write(best_score)
    #st.caption('Combination of hyperparameters for this is: ')
    #st.write(opt.results.best_params_)

with col2:
    st.header('Plot from the experiment: ')
    with st.form('my_form'):
        options = st.multiselect(
            'Which columns do you want in the plot?',
            df.columns.to_list(),
            ['Adjusted R^2 Score']
        )
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.line_chart(df, y=options)

