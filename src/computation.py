import numpy as np
import lmfit
from copy import deepcopy
import pandas as pd


from src import config as cfg
from src import helper_functions as hf
#import xarray as xr


#####################################

def get_bounds_dict(min_t_units, max_t_units):
    bounds_dict = deepcopy(cfg.bounds_dict
        )
    for parameter_name, bound in bounds_dict.items():
        if bound is None:
            bounds_dict[parameter_name] = (min_t_units, max_t_units)
    return bounds_dict


def create_vectors(parameters, param_names_list, min_t_units=0, max_t_units = 7) -> pd.DataFrame:

    v=parameters.valuesdict()
    assert(cfg.interp_type in ('linear', 'spline'))
           
    num_timesteps = hf.get_num_timesteps(max_t_units, min_t_units)
    dt = hf.get_dt(max_t_units, min_t_units)

    timesteps = (np.arange(0,num_timesteps)*dt )+ min_t_units
    zero_index = round(-1*min_t_units/dt)

    out_dict = {}
    out_dict['timesteps'] = timesteps
    for name in param_names_list:
        x_vect = []
        y_vect = []
        i=0
        while True:
            try:
                x_vect.append(v[f'{name}_tp_{i}'])
                y_vect.append(v[f'{name}_{i}'])
                i+=1
            except KeyError as E:
                break     
        if cfg.interp_type == 'linear':
            out_dict[name] = np.interp(timesteps, x_vect, y_vect)
        elif cfg.interp_type == 'spline':
            cs = CubicSpline(x_vect, y_vect,
                     bc_type = 'clamped')
            out_dict[name] = cs(timesteps)

    #these are all the same length so can be put in a dataframe
    time_varying_params_df = pd.DataFrame(out_dict)

    return time_varying_params_df


def simulate(parameters, param_names_list, min_t_units=0, max_t_units = 7) -> pd.DataFrame:

    time_varying_params_df = create_vectors(parameters, param_names_list, min_t_units=min_t_units, max_t_units = max_t_units)
    out_df = simulate_from_df(time_varying_params_df)
    return out_df


def simulate_from_df(time_varying_params_df) -> pd.DataFrame:
    timesteps = time_varying_params_df['timesteps']
    dt = np.mean(np.diff(timesteps))

    #this unpacking should cover all the fitable parameters, ^^timesteps is created during the simulation
    transcription_rate = time_varying_params_df['transcription_rate']
    splice_rate = time_varying_params_df['splice_rate']
    circ_rate = time_varying_params_df['circ_rate']
    lin_decay = time_varying_params_df['lin_decay']
    circ_decay = time_varying_params_df['circ_decay']

    assert(cfg.circ_downstream in ('both', 'yes', 'no'))
    if cfg.circ_downstream == 'both':
        homer1a_to_circ_switch = 1
        presplice_to_circ_switch = 1
    if cfg.circ_downstream == 'yes':
        homer1a_to_circ_switch = 1
        presplice_to_circ_switch = 0
    if cfg.circ_downstream == 'yes':
        homer1a_to_circ_switch = 0
        presplice_to_circ_switch = 1
        
    pre_splice_homer = np.zeros(timesteps.shape)
    homer1a = np.zeros(timesteps.shape)
    circ_homer = np.zeros(timesteps.shape)

    for i, t in enumerate(timesteps[1:]):
        converted_to_1a = splice_rate[i]*pre_splice_homer[i]*dt
        rest_to_circ = pre_splice_homer[i]*dt*circ_rate[i]*presplice_to_circ_switch
        homer1a_to_circ = homer1a[i]*dt*circ_rate[i]*homer1a_to_circ_switch

        delta_homer = transcription_rate[i] - converted_to_1a  - rest_to_circ - lin_decay[i]*dt*pre_splice_homer[i]
        delta_homer1a = converted_to_1a - homer1a_to_circ - lin_decay[i]*dt*homer1a[i]
        delta_circular = rest_to_circ + homer1a_to_circ - circ_decay[i]*dt*circ_homer[i]
        
        pre_splice_homer[i+1] = pre_splice_homer[i]+delta_homer
        homer1a[i+1] = homer1a[i]+delta_homer1a
        circ_homer[i+1] = circ_homer[i]+delta_circular
    
    #normalized_linear_amount = homer1a/(homer1a[zero_index])
    #normalized_circular_amount = circ_homer/(circ_homer[zero_index])
    #no longer doing this here so we don't lose information, easy to do later

    out_dict = {}
    for name in time_varying_params_df.columns:
        out_dict[name] = time_varying_params_df[name]
    out_dict['pre_splice_homer1_amount'] = pre_splice_homer
    out_dict['linear_homer1a_amount'] = homer1a
    out_dict['circ_homer1_amount'] = circ_homer

    #these are all the same length so can be put in a dataframe
    out_df = pd.DataFrame(out_dict)
    
    return out_df


def normalize_to_t0(timepoints, values_at_times):
    #print(len(timepoints))
    #print(len(values_at_times))
    zero_index = np.searchsorted(timepoints, [0])[0]
    return values_at_times/values_at_times[zero_index]


def error_func(time_varying_simulated_amounts_df, targets_df):
    timesteps = np.array(time_varying_simulated_amounts_df['timesteps'])
    measured_timepoints = targets_df['measured_timepoints']
    measured_homer1a_levels = targets_df['measured_homer1a_levels']
    measured_circ_homer_levels = targets_df['measured_circ_homer_levels']
    
    #print(time_varying_simulated_amounts_df.head(5))
    #print(timesteps)
    #print(time_varying_simulated_amounts_df['linear_homer1a_amount'])
    #normalize them all and subtract simulated from measured
    norm_sim_homer1a_timecourse = normalize_to_t0(timesteps, time_varying_simulated_amounts_df['linear_homer1a_amount'])
    simulated_1a_values = np.interp(measured_timepoints, timesteps, norm_sim_homer1a_timecourse)
    #print(simulated_1a_values)
    lin_1a_error = normalize_to_t0(measured_timepoints, measured_homer1a_levels) - simulated_1a_values
    
    norm_sim_circ_timecourse = normalize_to_t0(timesteps, time_varying_simulated_amounts_df['circ_homer1_amount'])
    simulated_circ_values = np.interp(measured_timepoints, timesteps, norm_sim_circ_timecourse)
    circ_error = normalize_to_t0(measured_timepoints, measured_circ_homer_levels) - simulated_circ_values
    #is there really any point in keeping these seperate? at one point I wanted to see which was causing more error
    
    #print(lin_1a_error)
    #print(circ_error)
    residuals = list(circ_error)
    residuals.extend(list(lin_1a_error))
    residuals.extend([0,0,0,0,0])
    residuals = np.array(residuals)
    #print(residuals)
    return residuals

    
def fit_system(params, param_names_list, targets_df):
    min_t = -cfg.simulate_t_units_before_t0
    max_t = cfg.get_max_t(targets_df['measured_timepoints'])
    
    def fcn2min(params, param_names_list, min_t_units, max_t_units, targets_df):
        #params.pretty_print()
        #print(param_names_list)
        #print(min_t_units)
        #print(max_t_units)
        #print(targets_df.head(5))
        #raise()
        time_varying_df = simulate(params, param_names_list, min_t_units=min_t, max_t_units=max_t)
        residuals = error_func(time_varying_df, targets_df)
        return residuals
    #optimize the paramters
    # do fit, here with the default leastsq algorithm
    minner = lmfit.Minimizer(fcn2min, params, fcn_args=(param_names_list, min_t, max_t, targets_df))
    result = minner.minimize()

    #rerun with the best parameters so we have the full timecourse of the simulation
    time_varying_df = simulate(result.params, param_names_list, min_t_units=min_t, max_t_units=max_t)
    
    # write error report
    #report_fit(result)

    #k=result.ndata
    #n=result.nfree
    #aic = result.aic
    #bic = result.bic
    #do my calculations match?
    #compute_aic_bic(residuals, k, n)
    return result, time_varying_df


#################################################################
#%%
def compute_rss_deprecated():
  squared_error_homer1a = np.sum((homer1a_predicted - norm_linear_homer_levels)**2)
  squared_error_circ = np.sum((circhomer_predicted - norm_circ_homer_levels)**2)
  rss = (squared_error_homer1a+squared_error_circ)
  if np.isnan(rss):
    rss = 100000000000 #just needs to be bigger than error for a reasonable set of parameters
  return rss

  def compute_aic_bic_depracated(residuals, k, n):
    rss = np.sum(residuals**2)
    aic = 2*k+n*np.log(rss/n)
    #aicc = aic + (2*k**2 + 2*k)/(n-k-1)
    #print(f'aicc: {aicc}')
    bic = n*np.log(rss/n) + k * np.log(n)
    return aic, bic
