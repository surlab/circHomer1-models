#intended to be functions that are used in the other src fils (computation.py, plotting.py) but avoids putting them in config.py
#each other module should import helper_functions as hf
import numpy as np
import lmfit
from copy import deepcopy

from src import config as cfg
from src import data_io as io


def get_num_timesteps(max_t_units, min_t_units):
    return (max_t_units-min_t_units)*cfg.steps_per_t_unit


def get_dt(max_t_units, min_t_units):
    return (max_t_units-min_t_units)/get_num_timesteps(max_t_units, min_t_units) #also should be 1/cfg.steps_per_t_unit?


def get_bounds_dict(min_t_units, max_t_units):
	bounds_dict = deepcopy(cfg.bounds_dict
		)
	for parameter_name, bound in bounds_dict.items():
	    if bound is None:
	        bounds_dict[parameter_name] = (min_t_units, max_t_units)
	return bounds_dict


def create_new_homer_parameters(timepoints: [float], param_names_list) -> lmfit.Parameters:
    params = lmfit.Parameters()
    #for each change timepoint, we want to add values for each of the 5 basic parameters: 
    #transcription rate, homer1a spice rate, circularizing rate, linear decay rate and circular decay rate
    
    for name in param_names_list:
        for i, timepoint in enumerate(timepoints):
            if i==0:             
                params.add(f'{name}_{i}', value=0, vary=True)
                params.add(f'{name}_tp_{i}', value=timepoint, vary=False)
            else:
                params.add(f'{name}_{i}', expr=f'{name}_0')
                params.add(f'{name}_tp_{i}', value=timepoint, vary=False)
                #if we want it to be a change and not a bump, then we can set them all equal to the previous instead of 0, but would need to do something about timepoints too. lets just see if this works 
    return params


def update_params(params_in, new_params_dict, bounds_dict) -> lmfit.Parameters:
    new_params = deepcopy(params_in)
    for name, set_params in new_params_dict.items():
        for i, value in enumerate(set_params):
            if type(value) is str:
                new_params[f'{name}_{i}'] = lmfit.Parameter(
                                                    f'{name}_{i}', 
                                                    expr = value
                                                    )
            else:
                new_params[f'{name}_{i}'] = lmfit.Parameter(
                                                    f'{name}_{i}', 
                                                    value=value, 
                                                    vary=True, 
                                                    min=bounds_dict[name][0], 
                                                    max=bounds_dict[name][1]
                                                    )
    return new_params


