from flask import Flask, request
from waitress import serve
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import requests
import psutil
import string
import sys
import os
import re
import json
import random
import socket
import subprocess
import pandas as pd

tempPath = os.path.join(os.path.dirname(__file__), 'temp')
opt_id = str(random.randint(1000000000,9999999999))

engine = create_engine("sqlite:///data.db")

global experiments
experiments = pd.DataFrame()

# Creating database functions
def insert_experiment(id, experiment_name, experiment: pd.DataFrame):
    #experiment.to_sql(f"{experiment_name}_{id}", engine, if_exists="append", index=False)
    global experiments
    experiments = pd.concat([experiment], experiments)

#def get_experiment(id, experiment_name):
    #query = f"SELECT * from {experiment_name}_{id}"
   
    #try: 
    #    df = pd.read_sql(query, engine)
    #except OperationalError as e:
    #    df = pd.DataFrame()
    #global experiments
    #return experiments

def experiment_exists(id, experiment_name):
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    df = pd.read_sql_query(query, engine)
    if f'{experiment_name}_{id}' in df['name'].values:
        return True
    else:
        return False

# Create a Flask application
app = Flask(__name__)

# Create landing page to confirm the server is running
@app.route('/')
def landing_page():
    return "Server is up!"


@app.route('/api/dashboards', methods=['GET'])
def show_active_dashboard():
    active_dashboards = active_dashboard()
    if active_dashboards == None: return 'No dashboards running'
    else: return str(active_dashboards)
    

@app.route('/api/data', methods=['GET'])
def get_data():
    #data_json = get_experiment(opt_id, 'experiment').to_json()
    #print(data_json)
    return {}


@app.route('/api/data', methods=['POST'])
def receive_data():
    # Retrieve the data from the request
    data = request.data.decode("utf-8")
    # Process the data...
    write_experiment_to_file(data)
    #insert_experiment(opt_id, 'experiment', pd.read_json(data))
    
    # Return a response to the client
    return 'Success!'


# Write experiments to disk
def write_experiment_to_file(json_object):
    """Writes an experiment result in json format to a file"""
    
    tempfile = opt_id + ''.join(random.choice(string.ascii_letters) for i in range(10))
    filePath = os.path.join(tempPath, tempfile)

    with open(filePath + ".json", "w") as outfile:
        outfile.write(json_object)


# Find your ip and an open port
def get_ip_and_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    _port = s.getsockname()[1]
    s.close()

    hostname = socket.gethostname()
    _ip = socket.gethostbyname(hostname)
    return _ip, _port


# Get active dashboards
def active_dashboard():
    for proc in psutil.process_iter():
        try:
            # Get the command-line arguments of the process
            cmdline = proc.cmdline()
            
            # Check if the process is the correct one
            if 'streamlit' in cmdline and 'run' in cmdline and opt_id in cmdline:
                return proc.pid
        except:
            pass
    
    return None


# Start dashboard
def start_dashboard(api_url):
    python_executable = sys.executable
    script_folder = os.path.dirname(__file__)
    script_path = os.path.join(script_folder,'web_all.py')
    subprocess.Popen([python_executable, '-m', 'streamlit', 'run', script_path, opt_id, api_url])


# Start the Flask application
if __name__ == '__main__':
    ip, port = get_ip_and_open_port()
    api_url = f'http://{ip}:{port}'
    start_dashboard(api_url)
    
    print(f'API accessible at http://{ip}:{port}')
    print('Starting dashboard...')
    serve(app, port=port, threads=1)