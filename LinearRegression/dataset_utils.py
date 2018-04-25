###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import numpy as np
import random
import pandas as pd



def createDataset(a, b, jump, func):
    random.seed(100)
    x = np.array([])
    y = np.array([])
    while a < b:
        x = np.append(x, a)
        #y = np.append(y, 2*a + 1*random.uniform(0, 30))
        y = np.append(y, func(a))
        a += jump
    dfx = pd.DataFrame({'x': x})
    dfy = pd.DataFrame({'y': y})
    return dfx.join(dfy)

def predictDataset(a, b, jump, regressor):
    y = np.array([])
    while a < b:
        y = np.append(y, regressor.predict(a))
        a += jump
    return pd.DataFrame({'y': y})
