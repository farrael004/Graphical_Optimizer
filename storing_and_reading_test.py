import pandas as pd
import json
import os
import shutil
import random
import string

path = os.path.join(os.getcwd(), "temp")
try:
    shutil.rmtree(path)
except FileNotFoundError:
    pass
os.makedirs(path, exist_ok=True)
for i in range(10):
    dictionary = {"test1": i}
    json_object = json.dumps(dictionary, indent=4)
    tempfile = ''.join(random.choice(string.ascii_letters) for i in range(10))
    filePath = os.path.join(path, tempfile)

    with open(filePath+".json", "w") as outfile:
        outfile.write(json_object)

finalDictionary = pd.DataFrame()

for filename in os.listdir(path):
    tempfile = os.path.join(path, filename)
    with open(tempfile, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        results = pd.DataFrame(json_object, index=[0])
        finalDictionary = pd.concat([finalDictionary, results], ignore_index=True, axis=0)

shutil.rmtree(path)
print(finalDictionary)

