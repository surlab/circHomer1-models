import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import os
import seaborn as sns

from src import config as cfg
from src import computation as comp
from src import helper_functions as hf



def make_timecourse_figures(time_varying_df, targets_df, title='') -> tuple([matplotlib.figure.Figure, plt.Axes]):  
    
    timesteps = np.array(time_varying_df['timesteps'])
    measured_timepoints = targets_df['measured_timepoints']

    fig, axs = plt.subplots(1, 5, figsize=(15,4))
    #first plot the trrascription
    axs[0].plot(timesteps, time_varying_df['transcription_rate'])
    axs[0].set_xlim(-cfg.display_t_units_before_t0, max(timesteps))
    axs[0].set_xlabel(f'Time ({cfg.t_units})')
    axs[0].set_ylabel('Transcription rate')
    
    #second the splice and circularizing rates
    axs[1].plot(timesteps, time_varying_df['splice_rate'], label='Homer1a_splice_rate')
    axs[1].plot(timesteps, time_varying_df['circ_rate'], label='Homer1_circularizing_rate')
    axs[1].set_xlim(-cfg.display_t_units_before_t0, max(timesteps))
    axs[1].set_xlabel(f'Time ({cfg.t_units})')
    axs[1].set_ylabel('Rate Value')
    axs[1].legend(loc=6, prop={'size': 8})
    
    #third the decay rates... This is going to have to wait, actually need to simulate the decay
    #after fetching decay values from result or time_varying_df
    #simulating it might be safest way to get the units right, but then I need to make a whole new base params and override the transcription rate to turn off at 0
    #axs[2].plot(t, s1, t, s2)
    #axs[2].set_xlim(-cfg.display_t_units_before_t0, 2)
    #axs[2].set_xlabel('Time (s)')
    #axs[2].set_ylabel('s1 and s2')
    #axs[2].legend(loc=1, prop={'size': 8})

    #4th the timecouses themselves
    homer1a_simulated = comp.normalize_to_t0(timesteps, time_varying_df['linear_homer1a_amount'])
    circ_homer_simulated = comp.normalize_to_t0(timesteps, time_varying_df['circ_homer1_amount'])
    axs[3].plot(timesteps, homer1a_simulated, label='Homer1a_simulated')
    axs[3].plot(timesteps, circ_homer_simulated, label='CircHomer_simulated')
    axs[3].scatter(measured_timepoints, targets_df['measured_homer1a_levels'], label='Homer1a_masured')
    axs[3].scatter(measured_timepoints, targets_df['measured_circ_homer_levels'], label='CircHomer_measured')
    axs[3].set_xlim(-cfg.display_t_units_before_t0, max(timesteps))
    axs[3].set_xlabel(f'Time ({cfg.t_units})')
    axs[3].set_ylabel('Transcript amount (relative to T0)')
    axs[3].legend(loc=1, prop={'size': 8})

    #the the ratio between the two transcripts
    axs[4].plot(timesteps, circ_homer_simulated/homer1a_simulated, label='simulated_ratio')
    axs[4].scatter(measured_timepoints, targets_df['measured_circ_homer_levels']/targets_df['measured_homer1a_levels'], label='measured_ratio')
    axs[4].set_xlim(-cfg.display_t_units_before_t0, max(timesteps))
    axs[4].set_xlabel(f'Time ({cfg.t_units})')
    axs[4].set_ylabel('Ratio of CircHomer to Homer1a (Relative to T0)')
    axs[4].legend(loc=1, prop={'size': 8})

    fig.suptitle(title)
    return fig, axs, circ_homer_simulated/homer1a_simulated



def make_stat_bar_chart(names, stat_list, stat_name='aic', title='') -> tuple([matplotlib.figure.Figure, plt.Axes]):  
    
    fig, ax = plt.subplots()

    positive_stat_list = stat_list-min(stat_list)+1
    norm_stat_list = positive_stat_list/max(positive_stat_list)
    ax.bar(names, norm_stat_list)
    ax.set_xlabel('Model Specification')
    ax.set_ylabel(f'Normalized {stat_name} (lower is better)')

    fig.suptitle(title)
    return (fig, ax)


def make_overlapping_ratio_timecourse_plot(targets_df, time_varying_df, ratio_timecourses, title=''):

    fig, ax = plt.subplots()

    timesteps = np.array(time_varying_df['timesteps'])
    measured_timepoints = targets_df['measured_timepoints']
    for model_name, ratio_timecourse in ratio_timecourses.items():
        ax.plot(timesteps, ratio_timecourse, label=model_name)
    ax.scatter(measured_timepoints, targets_df['measured_circ_homer_levels']/targets_df['measured_homer1a_levels'], label='measured_ratio')
    ax.set_xlim(-cfg.display_t_units_before_t0, max(timesteps))
    ax.set_xlabel(f'Time ({cfg.t_units})')
    ax.set_ylabel('Ratio of CircHomer to Homer1a (Relative to T0)')
    ax.legend(loc=2, prop={'size': 12})

    fig.suptitle(title)
    return fig, ax