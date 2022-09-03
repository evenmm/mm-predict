# Functions for Mauricio. By Even MM 31.08.2022
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import pickle
from scipy import optimize
from scipy.optimize import least_squares
import scipy.stats
from copy import deepcopy
import time
import warnings
from multiprocessing import Pool
import seaborn as sns

# This function calculates the negative loglikelihood value for a normal loglikelihood. You must probably change a lot:
def negative_loglikelihood_any_model(parameters_x, args):
    observations, other_args = args
    predictions = "your code here or in separate function to make predictions given the parameters"
    sumofsquares = np.sum((observations - predictions)**2)
    placeholder_sigma = 1 # If fixed. Otherwise include as the last element in parameters_x and use the last line here instead
    negative_loglikelihood = (len(observations)/2)*np.log(2*np.pi*placeholder_sigma**2) + sumofsquares/(2*placeholder_sigma**2)
    #negative_loglikelihood = (len(observations)/2)*np.log(2*np.pi*parameters_x[-1]**2) + sumofsquares/(2*parameters_x[-1]**2)
    return negative_loglikelihood

# Now define starting x0, observations and other variables to pass to negative lkoglikelihood, and bounds, 
x0, observations, other_args, all_bounds = args
res = optimize.minimize(fun=negative_loglikelihood_any_model, x0=x0, args=(observations, other_args), bounds=all_bounds, options={'disp':False}, method='SLSQP') # L-BFGS-B chosen automatically with bounds


# Extra: Parallell processing
# I included these two just in case you need optimization speed. This does parallell optimization with 10k different starting points.
def get_optimization_result_any_model(args):
    x0, patient, all_bounds = args
    res = optimize.minimize(fun=negative_loglikelihood_any_model, x0=x0, args=(patient), bounds=all_bounds, options={'disp':False}, method='SLSQP') # L-BFGS-B chosen automatically with bounds
    return res

def estimate_drug_response_parameters_any_model(patient, lb, ub, N_iterations=10000):
    all_bounds = tuple([(lb[ii], ub[ii]) for ii in range(len(ub))])
    all_random_samples = np.random.uniform(0,1,(N_iterations, len(ub)))
    x0_array = lb + np.multiply(all_random_samples, (ub-lb))

    args = [(x0_array[i],patient,all_bounds) for i in range(len(x0_array))]
    with Pool(-1) as pool:
        optim_results = pool.map(get_optimization_result_any_model,args)
    fun_value_list = [elem.fun for elem in optim_results]
    min_f_index = fun_value_list.index(min(fun_value_list))
    x_value_list = [elem.x for elem in optim_results]
    best_x = x_value_list[min_f_index]

    parameters_x = Parameters(Y_0=best_x[0], pi_r=1-PI_LB, g_r=best_x[1], g_s=GROWTH_LB, k_1=0, sigma=best_x[2])
    return parameters_x

