# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/GreggHeller1/gregg-circ-homer-models/blob/main/scripts/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# + id="71ee021b"
#settings
# %load_ext autoreload
# %autoreload 2
try:
  import google.colab
  in_colab = True
except:
  in_colab = False
print(in_colab)
# + colab={"base_uri": "https://localhost:8080/"} id="4e02e926" outputId="84475a29-508b-4d96-adf5-e85665e994d2"
#installs (for colab only, run this once)
if in_colab:
    # ! git clone https://github.com/GreggHeller1/gregg-circ-homer-models.git
# + id="5e9731ca" editable=true slideshow={"slide_type": ""}
#local imports
#cwd if in colab for imports to work
if in_colab:
    # %cd /content/gregg-circ-homer-models
    
from src import data_io as io
from src import plotting as plot
from src import computation as comp
from src import helper_functions as hf
# + id="db51ef2e"
#imports
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure

#import xarray as xr
#import numpy as np
#import ipdb
#from PIL import Image #this needs to be after matplotlib??
#from scipy.stats import stats   
#import cProfile
#cProfile.run('out = comp.function(inputs)') #replace "function" and "inputs" to profile the function you want to optimize



# + colab={"base_uri": "https://localhost:8080/"} id="a06b6e4a" outputId="989c69e2-c8c4-43e0-9ba6-7a36f66be4c3" editable=true slideshow={"slide_type": ""}
#cwd if in colab for file loading to work
if in_colab:
    # %cd /content/gregg-circ-homer-models/scripts
    
test_path = os.path.join('demo_data', 'test.txt')
print(test_path)
cwd = os.getcwd()
repo_path = os.path.split(cwd)[0]

print(os.path.exists(test_path))

# -
#define paths Test that your data exists here
data_path = os.path.join(repo_path, 'demo_data', '3D Spine volume (um3).csv')
print(os.path.exists(data_path))


# + colab={"base_uri": "https://localhost:8080/"} id="b3586a50" outputId="56f159c6-3dbc-4b37-d217-083fb5d2e792"
#data inputs
# + id="82a5927b"
#data manipulation
# +
#Save data
# -

def main(data_path, prefix, x_label, scale_factor=1 ) -> tuple([matplotlib.figure.Figure, plt.Axes]):  
    #typehints from here https://stackoverflow.com/questions/43890844/pythonic-type-hints-with-pandas
    
    #Load data to make plots
    input_df = pd.read_csv(data_path) 
    
    #restructure df for plotting
    spine_measure_df = restructure_df(input_df, x_label)

    print(spine_measure_df.head(5))
    #rescale to account for expansion
    spine_measure_df[x_label] = spine_measure_df[x_label]/scale_factor

    #make plots
    fig, ax = plt.subplots()
    ax = sns.histplot(data=spine_measure_df, x=x_label, kde=True, log_scale=True, hue="source_experiment", element="step",
        stat="percent", common_norm=False,)

    #save plots
    save_path = os.path.join(repo_path, 'demo_data', f'{prefix}_histograms.png')
    fig.savefig(save_path, bbox_inches='tight')
    return (fig, ax)


# + editable=true slideshow={"slide_type": ""}
#Functions (to be pulled into different files later)

def restructure_df(df_in, x_label) -> pd.DataFrame:
    df_list = []
    for column in df_in.columns:
        for measure in df_in[column]:
            #if column == '0d sh-scramble':
            #    adjusted_volume = volume/4/1.7/2.5
            #else:
            #adjusted_volume = measure/(3.5**3)#/4/1.7
            row_dict = {
                'source_experiment': column,
                x_label: measure
            }
            df_list.append(row_dict)

    resturctured_df = pd.DataFrame(df_list)
    return resturctured_df



# -

data_path = os.path.join(repo_path, 'demo_data', '3D Spine volume (um3).csv')
print(os.path.exists(data_path))
main(data_path, 'volume', 'spine_volume_um3', scale_factor=3.5**3)



data_path = os.path.join(repo_path, 'demo_data', '1D Spine diameter (um).csv')
print(os.path.exists(data_path))
main(data_path, 'diameter', 'spine_diameter_um', scale_factor=3.5)




