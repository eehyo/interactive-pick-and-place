"""
init_ipython_setup.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image
from IPython import get_ipython

ipython = get_ipython()
if ipython is None:
    matplotlib.use("Agg")

print ("Basic packages initialized.")
print (" import numpy as np")
print (" import matplotlib.pyplot as plt")
print (" import scipy as sp")
print (" from PIL import Image")

# Inline commands for Jupyter Notebook
""" 
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
%gui qt
"""
if ipython:
    ipython.run_line_magic('config', 'InlineBackend.figure_format = "retina"')
    ipython.run_line_magic('matplotlib', 'inline')
    ipython.run_line_magic('gui', 'qt')
    print ("[ipython] settings (retina, inline, qt) initialized.")

# Numpy settings
np.set_printoptions(precision=2,suppress=True,linewidth=100)
np.seterr(divide="ignore",over="ignore",invalid="ignore")
print ("[numpy] settings initialized.")

# Matplotlib settings
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
print ("[matplotlib] settings initialized.")
