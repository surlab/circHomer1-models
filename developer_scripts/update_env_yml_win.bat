REM comment: this script updates the environment yml files, both an explicit and a cross platform  
cd ..
call conda activate gregg-circ-homer-models
call conda info --envs
call conda env export > environment_explicit.yml
call conda env export --from-history > environment_cross_platform.yml
cmd /k
