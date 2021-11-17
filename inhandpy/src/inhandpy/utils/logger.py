# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)

class Logger:
    def __init__(self, params=None):

        self.params = params 
        self.dataframe = pd.DataFrame()
    
    def get_data(self):
        return self.dataframe
    
    def log_val(self, names, vals, index_val, index_name=None):
        data_dict = {}
        for name, val in zip(names, vals):
            data_dict[name] = [val]
 
        data_row = pd.DataFrame(data_dict, index=[index_val], dtype=object)
        if index_name is not None: data_row.index.name = index_name
        dfs = [data_row] if self.dataframe.empty else [self.dataframe, data_row]

        self.dataframe = pd.concat([df.stack() for df in dfs], axis=0).unstack()

    def set_index(self, index_vals):
        self.dataframe = self.dataframe.set_index(index_vals)

    def write_data_to_file(self, csvfile, verbose=False):
        if verbose: log.info(f"Saving logged data to {csvfile}")
        self.dataframe.to_csv(csvfile)