---
jupyter:
  jupytext:
    formats: ipynb,py,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/GreggHeller1/gregg-circ-homer-models/blob/main/scripts/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->
```python id="71ee021b"
#settings
%load_ext autoreload
%autoreload 2
try:
  import google.colab
  in_colab = True
except:
  in_colab = False
print(in_colab)
```
```python colab={"base_uri": "https://localhost:8080/"} id="4e02e926" outputId="84475a29-508b-4d96-adf5-e85665e994d2"
#installs (for colab only, run this once)
if in_colab:
    ! git clone https://github.com/GreggHeller1/gregg-circ-homer-models.git
```
```python id="5e9731ca"
#local imports
#cwd if in colab for imports to work
if in_colab:
    %cd /content/gregg-circ-homer-models
    
from src import data_io as io
from src import plotting as plot
from src import computation as comp
from src import helper_functions as hf
```
```python id="db51ef2e"
#imports
import pandas as pd
import os

#import xarray as xr
#import numpy as np
#import ipdb
#from matplotlib import pyplot as plt
#from PIL import Image #this needs to be after matplotlib??
#from scipy.stats import stats   
import os
#import cProfile
#cProfile.run('out = comp.function(inputs)') #replace "function" and "inputs" to profile the function you want to optimize

```


```python colab={"base_uri": "https://localhost:8080/"} id="a06b6e4a" outputId="989c69e2-c8c4-43e0-9ba6-7a36f66be4c3"
#cwd if in colab for file loading to work
if in_colab:
    %cd /content/gregg-circ-homer-models/scripts
    
test_path = os.path.join('demo_data', 'test.txt')
print(test_path)
print(os.getcwd())
print(os.path.exists(test_path))

```
```python
#define paths Test that your data exists here
data_path = "/Users/Gregg/code/gregg-circ-homer-models/scripts/demo_data/ASC26_cell_3_soma.mat"
print(os.path.exists(soma_path))
```


```python colab={"base_uri": "https://localhost:8080/"} id="b3586a50" outputId="56f159c6-3dbc-4b37-d217-083fb5d2e792"
#data inputs

```


```python id="82a5927b"
#data manipulation
```

```python
#Save data
```

```python
#Load data to make plots
```
```python

```

