from flask import Flask, request
from waitress import serve
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

tempPath = os.path.join(os.path.dirname(__file__), 'temp')
opt_id = str(random.randint(1000000000,9999999999))

# Create a Flask application
app = Flask(__name__)

# Create landing page to confirm the server is running
@app.route('/')
def landing_page():
    return "Server is up!"


# Define a route that handles GET requests
@app.route('/api/dashboards', methods=['GET'])
def show_active_dashboard():
    active_dashboards = active_dashboard()
    if active_dashboards == None: return 'No dashboards running'
    else: return str(active_dashboards)


# Define a route that handles POST requests
@app.route('/api/data', methods=['POST'])
def receive_data():
    # Retrieve the data from the request
    data = request.data.decode("utf-8")
    
    # Process the data...
    #json_object = json.loads(data.decode('utf-8'))
    print(data)
    #write_experiment_to_file(data)
    
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
def start_dashboard():
    python_executable = sys.executable
    script_folder = os.path.dirname(__file__)
    script_path = os.path.join(script_folder,'main.py')
    subprocess.Popen([python_executable, '-m', 'streamlit', 'run', script_path, opt_id])

# Start the Flask application
if __name__ == '__main__':
    #start_dashboard()
    ip, port = get_ip_and_open_port()
    print(f'API accessible at http://{ip}:{port}')
    print('Starting dashboard...')
    serve(app, port=port, threads=1)
