from collections import OrderedDict

import numpy as np

class LogBuffer(object):

    def __init__(self, average_filter):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False
        self.average_filter = average_filter

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, val in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(val)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n value or all values"""
        assert n >= 0
        for key in self.val_history:
            if key in self.average_filter:
                continue
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True

