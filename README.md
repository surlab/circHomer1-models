# circHomer1-models
Code used in supplementary figure 4 for modeling circHomer1 dynamics. 

## Installation instructions
1.	These instructions require:
2.	a recent Python distribution, preferably anaconda 
3.	an installation of git. 
4.	The correct environment yamls. 
#### For a windows machine:
1. Open a command prompt and run the following commands
```bash
cd ..\documents\code #or similar as relevant for your machine
git clone git@github.com:surlab/circHomer1-models.git
```
2. Double click the file "user_install_win.bat" to run it. This should set up the conda environment and all dependencies. 
#### OR For a windows or non windows machine
Open a terminal and run the following commands
```bash
cd ..\documents\code #or similar as relevant for your machine
git clone git@github.com:surlab/circHomer1-models.git
cd circHomer1-models
conda env create -f environment_cross_platform.yml
Conda activate circHomer1-models
call pip install -e .
```
The installation should now be complete and the circHomer1-models conda environment should still be activated. 
## Usage instructions
1. make a copy of default_config.py and name it config.py.
1. change the path in config.py to a data directory containing the appropriate input files defined below
#### For a windows machine:
Double click the file "main.bat" to run it. 
#### OR for a windows or non windows machine
Open a terminal and run the following commands
```bash
cd path/to/circHomer1-models
conda activate circHomer1-models
python scripts/main.py
```
This code was created for the surlab at MIT by Gregg Heller.  
