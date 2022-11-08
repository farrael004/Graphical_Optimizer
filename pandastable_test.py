from tkinter import *
from pandastable import Table, TableModel, config
import threading
import pandas as pd

class _App(Frame, threading.Thread):
    def __init__(self, parent=None):
        self.parent = parent
        self.isUpdatingTable = True
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def run(self):
        Frame.__init__(self)
        self.main = self.master
        self.main.geometry('600x400+200+100')
        self.main.title('Optimizer')
        self.f = Frame(self.main)
        self.f.pack(fill=BOTH, expand=1)
        df = pd.DataFrame({"test": [[1,2,3,4,5],[2,2,3,4,5],[3,2,3,4,5]]})
        
        data = df
        if type(data.iloc[0].values[0]) is list:
            columns = data.columns.tolist()
            if len(columns) == 1:
                columns = columns[0]
            else:
                raise Exception("Cannot multiplot from more than one column.")

            data = pd.DataFrame(data[columns].tolist(), index=data.index.tolist())
            data = data.transpose()
            data.columns = [f'Experiment {i}' for i in data.columns.tolist()]

        
        self.table = pt = Table(self.f, dataframe=data,
                                showtoolbar=True, showstatusbar=True)
        pt.show()
        options = {'colheadercolor': 'green', 'floatprecision': 5}  # set some options
        config.apply_options(options, pt)
        pt.show()

        self.mainloop()

app = _App()