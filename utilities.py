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
#from drug_colors import *
#import seaborn as sns
import arviz as az
import pymc as pm
import gc # Garbage collector

def isNaN(string):
    return string != string
def Sort(sub_li): # Sorts a list of sublists on the second element in the list 
    return(sorted(sub_li, key = lambda x: x[1]))
def find_max_time(measurement_times):
    # Plot until last measurement time (last in array, or first nan in array)
    if np.isnan(measurement_times).any():
        last_time_index = np.where(np.isnan(measurement_times))[0][0] -1 # Last non-nan index
    else:
        last_time_index = -1
    return int(measurement_times[last_time_index])
def print_map(result):
    return pd.Series({k: v for k, v in result.items()})

s = 25 # scatter plot object size
GROWTH_LB = 0.001
PI_LB = 0.001
# Bounds for model 1: 1 population, only resistant cells
#                Y_0,  g_r  sigma
lb_1 = np.array([  0, GROWTH_LB, 10e-6])
ub_1 = np.array([100, 0.20, 10e4])

# Bounds for model 2: 1 population, only sensitive cells
#                Y_0,   g_s,    k_1,  sigma
lb_2 = np.array([  0,   GROWTH_LB,  0.20, 10e-6])
ub_2 = np.array([100, lb_2[2],  1.00, 10e4])

# Bounds for model 3: sensitive and resistant population
#                Y_0,       pi_r,    g_r,    g_s,     k_1,  sigma
lb_3 = np.array([  0,      PI_LB,  GROWTH_LB,   GROWTH_LB,   0.20, 10e-6])
ub_3 = np.array([100,    1-PI_LB,  0.20,  lb_3[4], 1.00, 10e4])
# Suggested from the server, as more informative: 
#lb_3 = np.array([  0,      PI_LB,  GROWTH_LB,   GROWTH_LB,   0.002, 10e-6])
#ub_3 = np.array([100,    1-PI_LB,  0.20,  lb_3[4], 0.2, 10e4])

np.random.seed(42)
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))
## Later, you want to restart the stream
#rs = RandomState(MT19937(SeedSequence(987654321)))

# Assumptions and definitions: 
# The atomic unit of time is 1 day
# Treatment lines must be back to back: Start of a treatment must equal end of previous treatment
# Growth rates are in unit 1/days
# 22.03.22 treatment in terms of different drugs

# In drug dictionary, key is drug name and value is drug id
treatment_id_to_drugs_dictionary = {}
drug_dictionary_OSLO = {}
drug_dictionary = {}
#drug_dictionary_OSLO = np.load("./binaries_and_pickles/drug_dictionary_OSLO.npy", allow_pickle=True).item()
#drug_dictionary_COMMPASS = np.load("./binaries_and_pickles/drug_dictionary_COMMPASS.npy", allow_pickle=True).item()
#for key, value in drug_dictionary_COMMPASS.items():
#    if key not in drug_dictionary_OSLO.keys():
#        drug_dictionary_OSLO[key] = (max(drug_dictionary_OSLO.values())+1)
# Join the two drug dictionaries to get a complete drug dictionary 
#drug_dictionary = drug_dictionary_OSLO
#np.save("./binaries_and_pickles/drug_dictionary.npy", drug_dictionary)
#drug_id_to_name_dictionary = {v: k for k, v in drug_dictionary.items()}
#treatment_to_id_dictionary = np.load("./binaries_and_pickles/treatment_to_id_dictionary_OSLO.npy", allow_pickle=True).item()
##treatment_to_id_dictionary = np.load("./binaries_and_pickles/treatment_to_id_dictionary_COMMPASS.npy", allow_pickle=True).item()
#treatment_id_to_drugs_dictionary = {v: k for k, v in treatment_to_id_dictionary.items()}

#def get_drug_names_from_treatment_id_COMMPASS(treatment_id, treatment_id_to_drugs_dictionary_COMMPASS):
#    drug_set = treatment_id_to_drugs_dictionary_COMMPASS[treatment_id]
#    drug_names = [drug_id_to_name_dictionary[elem] for elem in drug_set]
#    return drug_names

#unique_drugs = ['Revlimid (lenalidomide)', 'Cyclophosphamide', 'Pomalidomide',
# 'Thalidomide', 'Velcade (bortezomib) - subcut twice weekly',
# 'Dexamethasone', 'Velcade (bortezomib) - subcut once weekly', 'Carfilzomib',
# 'Melphalan', 'Panobinostat', 'Daratumumab', 'Velcade i.v. twice weekly',
# 'Bendamustin', 'Methotrexate i.t.', 'Prednisolon', 'pembrolizumab',
# 'Pembrolizumab', 'Doxorubicin', 'Velcade i.v. once weekly', 'Vincristine',
# 'Ibrutinib', 'lxazomib', 'Cytarabin i.t.', 'Solu-Medrol',
# 'Velcade s.c. every 2nd week', 'Clarithromycin', 'hydroxychloroquine',
# 'Metformin', 'Rituximab']
#drug_ids = range(len(unique_drugs))
#drug_dictionary_OSLO = dict(zip(unique_drugs, drug_ids))
#unique_treatment_lines = []
#drug_ids = range(len(unique_treatment_lines))
#treatment_to_id_dictionary = dict(zip(unique_treatment_lines, treatment_line_ids))

#####################################
# Classes, functions
#####################################

class Cell_population:
    # alpha is the growth rate with no drug
    # k_d is the additive effect of drug d on the growth rate
    def __init__(self, alpha, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9):
        self.alpha = alpha
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.k_5 = k_5
        self.k_6 = k_6
        self.k_7 = k_7
        self.k_8 = k_8
        self.k_9 = k_9
        self.drug_effects = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9]
    def get_growth_rate(self, treatment):
        this_drug_array = treatment.get_drug_ids()
        drug_effect_filter = [drug_id in this_drug_array for drug_id in range(len(self.drug_effects))]
        print(drug_effect_filter)
        return self.alpha - sum(self.drug_effects[drug_effect_filter])

class Parameters: 
    def __init__(self, Y_0, pi_r, g_r, g_s, k_1, sigma):
        self.Y_0 = Y_0 # M protein value at start of treatment
        self.pi_r = pi_r # Fraction of resistant cells at start of treatment 
        self.g_r = g_r # Growth rate of resistant cells
        self.g_s = g_s # Growth rate of sensitive cells in absence of treatment
        self.k_1 = k_1 # Additive effect of treatment on growth rate of sensitive cells
        self.sigma = sigma # Standard deviation of measurement noise
    def to_array_without_sigma(self):
        return np.array([self.Y_0, self.pi_r, self.g_r, self.g_s, self.k_1])
    def to_array_with_sigma(self):
        return np.array([self.Y_0, self.pi_r, self.g_r, self.g_s, self.k_1, self.sigma])
    def to_array_for_prediction(self):
        return np.array([self.pi_r, self.g_r, (self.g_s - self.k_1)])
    def to_prediction_array_composite_g_s_and_K_1(self):
        return [self.pi_r, self.g_r, (self.g_s - self.k_1)]

class Treatment:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id
    def get_drug_ids(self):
        return [drug_dictionary_OSLO[drug_name] for drug_name in treatment_id_to_drugs_dictionary[self.id]]

class Drug_period:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id

class Patient: 
    def __init__(self, parameters, measurement_times, treatment_history, covariates = [], name = "nn"):
        self.measurement_times = measurement_times
        self.treatment_history = treatment_history
        self.Mprotein_values = measure_Mprotein_with_noise(parameters, self.measurement_times, self.treatment_history)
        self.covariates = covariates
        self.name = name
        self.longitudinal_data = {} # dictionary where the keys are covariate_names, the entries numpy arrays! 
        self.longitudinal_times = {} # matching time arrays for each covariate
        self.pfs = []
        self.mrd_results = []
        self.mrd_times = []
    def get_measurement_times(self):
        return self.measurement_times
    def get_treatment_history(self):
        return self.treatment_history
    def get_Mprotein_values(self):
        return self.Mprotein_values
    def get_covariates(self):
        return self.covariates

class Real_Patient: 
    def __init__(self, measurement_times, Mprotein_values, covariates=[], name="no_id", pfs=np.nan):
        self.measurement_times = measurement_times # numpy array
        self.treatment_history = np.array([Treatment(start=1, end=measurement_times[-1], id=1)])
        self.Mprotein_values = Mprotein_values # numpy array
        self.covariates = covariates # Optional 
        self.name = name # id
        self.longitudinal_data = {} # dictionary where the keys are covariate_names, the entries numpy arrays! 
        self.longitudinal_times = {} # matching time arrays for each covariate
        self.pfs = pfs
        #self.KappaLambdaRatio = [np.nan for elem in measurement_times] # This is part of longitudinals
        self.mrd_results = []
        self.mrd_times = []
    def get_treatment_history(self):
        return self.treatment_history
    def get_Mprotein_values(self):
        return self.Mprotein_values
    def get_measurement_times(self):
        return self.measurement_times
    def get_covariates(self):
        return self.covariates
    def add_Mprotein_line_to_patient(self, time, Mprotein):
        self.measurement_times = np.append(self.measurement_times,[time])
        self.treatment_history = np.array([Treatment(start=1, end=time, id=1)])
        self.Mprotein_values = np.append(self.Mprotein_values,[Mprotein])
        return 0

class COMMPASS_Patient: 
    def __init__(self, measurement_times, drug_dates, drug_history, treatment_history, Mprotein_values, Kappa_values, Lambda_values, covariates, name):
        self.name = name # string: patient id
        self.measurement_times = measurement_times # numpy array: dates for M protein measurements in Mprotein_values
        self.Mprotein_values = Mprotein_values # numpy array: M protein measurements
        self.drug_history = drug_history # numpy array of possibly overlapping single Drug_period instances, a pre-step to make treatment_history
        self.drug_dates = drug_dates # set: set of drug dates that marks the intervals 
        self.treatment_history = treatment_history # numpy array of non-overlapping Treatment instances (drug combinations) sorted by occurrence
        self.covariates = covariates # ?? : other covariates
        self.parameter_estimates = np.array([])
        self.parameter_periods = np.array([])
        self.dummmy_patients = np.array([])
        self.Kappa_values = Kappa_values
        self.Lambda_values = Lambda_values
    def get_measurement_times(self):
        return self.measurement_times
    def get_treatment_history(self):
        return self.treatment_history
    def get_drug_history(self):
        return self.drug_history
    def get_Mprotein_values(self):
        return self.Mprotein_values
    def print(self):
        print("\nPrinting patient "+self.name)
        print("M protein values:")
        print(self.Mprotein_values)
        print("Measurement times:")
        print(self.measurement_times)
        print("Drug history:")
        for number, drug_period in enumerate(self.drug_history):
            print(str(number) + ": [",drug_period.start, "-", drug_period.end, "]: drug id", drug_period.id)
        print("Treatment history:")
        for number, treatment in enumerate(self.treatment_history):
            print(str(number) + ": [",treatment.start, "-", treatment.end, "]: treatment id", treatment.id)
        if len(self.parameter_estimates) > 0:
            print("Estimated parameters by period:")
            for number, parameters in enumerate(self.parameter_estimates):
                print(parameters.to_array_without_sigma(), "in", self.parameter_periods[number])
    def add_Mprotein_line_to_patient(self, time, Mprotein, Kappa, Lambda):
        self.measurement_times = np.append(self.measurement_times,[time])
        self.Mprotein_values = np.append(self.Mprotein_values,[Mprotein])
        self.Kappa_values = np.append(self.Kappa_values,[Kappa])
        self.Lambda_values = np.append(self.Lambda_values,[Lambda])
        return 0
    def add_drug_period_to_patient(self, drug_period_object):
        self.drug_history = np.append(self.drug_history,[drug_period_object])
        self.drug_dates.add(drug_period_object.start)
        self.drug_dates.add(drug_period_object.end)
        return 0
    def add_treatment_to_treatment_history(self, treatment_object):
        self.treatment_history = np.append(self.treatment_history,[treatment_object])
        return 0
    def add_parameter_estimate(self, estimates, period, dummmy_patient):
        #if len(self.parameter_estimates) == 0:
        #    self.parameter_estimates = np.zeros(len(self.treatment_history))
        self.parameter_estimates = np.append(self.parameter_estimates,[estimates])
        if len(self.parameter_periods) == 0:
            self.parameter_periods = np.array([period]) 
            self.dummmy_patients = np.array([dummmy_patient])
        else:
            self.parameter_periods = np.append(self.parameter_periods, np.array([period]), axis=0)
            self.dummmy_patients = np.append(self.dummmy_patients, np.array([dummmy_patient]), axis=0)

class Filter:
    def __init__(self, filter_type, bandwidth, lag):
        self.filter_type = filter_type # flat or gauss
        self.bandwidth = bandwidth # days
        self.lag = lag # days

def compute_drug_filter_values(this_filter, patient, end_of_history): # end_of_history: The time at which history ends and the treatment of interest begins 
    # Returns a dictionary with key: drug_id (string) and value: filter feature value (float) for all drugs 
    filter_end = max(0, end_of_history - this_filter.lag)
    filter_start = max(0, end_of_history - this_filter.lag - this_filter.bandwidth)
    # Loop through drug periods in patient history and update the filter values by computing overlap with filter interval 
    #filter_value_dictionary = {k: 0 for k, v in drug_dictionary.items()}
    filter_values = [0 for ii in range(len(drug_dictionary))]
    for drug_period in patient.drug_history:
        # Overlap = min(ends) - max(starts)
        overlap = min(drug_period.end, filter_end) - max(drug_period.start, filter_start)
        # Set 0 if negative
        filter_addition = max(0, overlap)
        #filter_value_dictionary[drug_period.id] = filter_value_dictionary[drug_period.id] + filter_addition
        filter_values[drug_period.id] = filter_values[drug_period.id] + filter_addition
    return filter_values
    # This function is vectorized w.r.t different filters only, not drug ids or drug periods because two drug periods can have the same drug id 

def compute_filter_values(this_filter, patient, end_of_history, value_type="Mprotein"): # end_of_history: The time at which history ends and the treatment of interest begins 
    # Returns a dictionary with key: drug_id (string) and value: filter feature value (float) for all drugs 
    filter_end = max(0, end_of_history - this_filter.lag)
    filter_start = max(0, end_of_history - this_filter.lag - this_filter.bandwidth)
    correct_times = (patient.measurement_times >= filter_start) & (patient.measurement_times <= filter_end)
    if value_type == "Mprotein":
        type_values = patient.Mprotein_values[correct_times]
    elif value_type == "Kappa":
        type_values = patient.Kappa_values[correct_times]
    elif value_type == "Lambda":
        type_values = patient.Lambda_values[correct_times]
    # Automatically handling empty lists since sum([]) = 0.
    if this_filter.filter_type == "flat":
        filter_values = [sum(type_values)]
    elif this_filter.filter_type == "gauss":
        # For Gaussian filters, lag = offset and bandwidth = variance
        values = type_values
        time_deltas = patient.measurement_times[correct_times] - filter_end
        gauss_weighting = scipy.stats.norm(filter_end, this_filter.bandwidth).pdf(time_deltas)
        filter_values = [sum(gauss_weighting * values)]
    return filter_values

#####################################
# Generative models, simulated data
#####################################
# Efficient implementation 
# Simulates M protein value at times [t + delta_T]_i
# Y_t is the M protein level at start of time interval
def generative_model(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values) + Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

def generate_resistant_Mprotein(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values)

def generate_sensitive_Mprotein(Y_t, params, delta_T_values, drug_effect):
    return Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

def get_pi_r_after_time_has_passed(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times, dtype=float) #without dtype you get ints
    # Adding a small epsilon to Y and pi_r to improve numerical stability
    epsilon_value = 1e-15
    Y_t = params.Y_0# + epsilon_value
    pi_r_t = params.pi_r# + epsilon_value
    t_params = Parameters(Y_t, pi_r_t, params.g_r, params.g_s, params.k_1, params.sigma)
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        this_treatment = treatment_history[treat_index]
        if this_treatment.id == 0:
            drug_effect = 0
        #elif this_treatment.id == 1:
        # With inference only for individual combinations at a time, it is either 0 or "treatment on", which is k1
        else:
            drug_effect = t_params.k_1
        #else:
        #    sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        
        # Filter that selects measurement times occuring while on this treatment line
        correct_times = (measurement_times >= this_treatment.start) & (measurement_times <= this_treatment.end)
        
        delta_T_values = measurement_times[correct_times] - this_treatment.start
        # Add delta T for (end - start) to keep track of Mprotein at end of treatment
        delta_T_values = np.concatenate((delta_T_values, np.array([this_treatment.end - this_treatment.start])))

        # Calculate Mprotein values
        # resistant 
        resistant_mprotein = generate_resistant_Mprotein(Y_t, t_params, delta_T_values, drug_effect)
        # sensitive
        sensitive_mprotein = generate_sensitive_Mprotein(Y_t, t_params, delta_T_values, drug_effect)
        # summed
        recorded_and_endtime_mprotein_values = resistant_mprotein + sensitive_mprotein
        # Assign M protein values for measurement times that are in this treatment period
        Mprotein_values[correct_times] = recorded_and_endtime_mprotein_values[0:-1]
        # Store Mprotein value at the end of this treatment:
        Y_t = recorded_and_endtime_mprotein_values[-1]# + epsilon_value
        pi_r_t = resistant_mprotein[-1] / (resistant_mprotein[-1] + sensitive_mprotein[-1] + epsilon_value) # Add a small number to keep numerics ok
        t_params = Parameters(Y_t, pi_r_t, t_params.g_r, t_params.g_s, t_params.k_1, t_params.sigma)
    return Mprotein_values, pi_r_t

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_noiseless(params, measurement_times, treatment_history):
    Mprotein_values, pi_r_after_time_has_passed = get_pi_r_after_time_has_passed(params, measurement_times, treatment_history)
    return Mprotein_values

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_with_noise(params, measurement_times, treatment_history):
    # Return true M protein value + Noise
    noise_array = np.random.normal(0, params.sigma, len(measurement_times))
    noisy_observations = measure_Mprotein_noiseless(params, measurement_times, treatment_history) + noise_array
    # thresholded at 0
    return np.array([max(0, value) for value in noisy_observations])

# Pass a Parameter object to this function along with an numpy array of time points in days
def measure_Mprotein_naive(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times)
    Y_t = params.Y_0
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        if treatment_history[treat_index].id == 0:
            drug_effect = 0
        elif treatment_history[treat_index].id == 1:
            drug_effect = params.k_1
        else:
            print("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
            sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        # Calculate the M protein value at the end of this treatment line
        Mprotein_values = params.Y_0 * params.pi_r * np.exp(params.g_r * measurement_times) + params.Y_0 * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * measurement_times)        
    return Mprotein_values
    #return params.Y_0 * params.pi_r * np.exp(params.g_r * measurement_times) + params.Y_0 * (1-params.pi_r) * np.exp((params.g_s - params.k_1) * measurement_times)

def generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_local, P, get_expected_theta_from_X, true_omega, true_omega_for_psi, seed, RANDOM_EFFECTS, DIFFERENT_LENGTHS=True, USUBJID=False, simulate_rho_r_dependancy_on_rho_s=False, coef_rho_s_rho_r=0, psi_population=50):
    np.random.seed(seed)
    #X_mean = np.repeat(0,P)
    #X_std = np.repeat(0.5,P)
    #X = np.random.normal(X_mean, X_std, size=(N_patients_local,P))
    X = np.random.uniform(-1, 1, size=(N_patients_local,P))
    X = pd.DataFrame(X, columns = ["Covariate "+str(ii+1) for ii in range(P)])
    expected_theta_1, expected_theta_2, expected_theta_3 = get_expected_theta_from_X(X, simulate_rho_r_dependancy_on_rho_s, coef_rho_s_rho_r)
    if USUBJID:
        X["USUBJID"] = ["Patient " + x for x in X.index.map(str)]

    # Set the seed again to make the random effects not change with P
    np.random.seed(seed+1)
    if RANDOM_EFFECTS:
        true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
        true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
        true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])
    else:
        true_theta_rho_s = expected_theta_1
        true_theta_rho_r = expected_theta_2
        true_theta_pi_r  = expected_theta_3

    # Set the seed again to get identical observation noise irrespective of random effects or not
    np.random.seed(seed+2)
    #true_theta_psi = np.random.normal(np.log(psi_population), true_omega_for_psi, size=N_patients_local)
    true_rho_s = - np.exp(true_theta_rho_s)
    true_rho_r = np.exp(true_theta_rho_r)
    true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
    #true_psi = np.exp(true_theta_psi)
    true_psi = np.random.uniform(5, 85, size=N_patients_local)

    # Set seed again to give patient random Numbers of M protein
    np.random.seed(seed+3)
    parameter_dictionary = {}
    patient_dictionary = {}
    for training_instance_id in range(N_patients_local):
        psi_patient_i   = true_psi[training_instance_id]
        pi_r_patient_i  = true_pi_r[training_instance_id]
        rho_r_patient_i = true_rho_r[training_instance_id]
        rho_s_patient_i = true_rho_s[training_instance_id]
        these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=true_sigma_obs)
        # Remove some measurement times from the end: 
        if DIFFERENT_LENGTHS:
            M_ii = np.random.randint(min(3,len(measurement_times)), len(measurement_times)+1)
        else:
            M_ii = len(measurement_times)
        measurement_times_ii = measurement_times[:M_ii]
        this_patient = Patient(these_parameters, measurement_times_ii, treatment_history, name=str(training_instance_id))
        patient_dictionary["Patient " + str(training_instance_id)] = this_patient
        # From server: 
        #patient_dictionary[training_instance_id] = this_patient
        parameter_dictionary[training_instance_id] = these_parameters
        #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id)
    return X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s
    #return X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r

#####################################
# Plotting
#####################################
#treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
def plot_mprotein(patient, title, savename, PLOT_PARAMETERS=False, parameters = [], PLOT_lightchains=False, plot_pfs=False, plot_KapLam=False):
    measurement_times = patient.measurement_times
    Mprotein_values = patient.Mprotein_values
    
    fig, ax1 = plt.subplots()
    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")
    if PLOT_lightchains:
        ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='b', label="Observed M protein")
    if plot_pfs:
        ax1.axvline(patient.pfs, color="r", linewidth=1, linestyle="--", label="PFS")
    if plot_KapLam:
        #ax1.plot(measurement_times, patient.KappaLambdaRatio, linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
        ax1.plot(patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")

    if PLOT_PARAMETERS:
        plotting_times = np.linspace(measurement_times[0], measurement_times[-1], 80)
        treatment_history = np.array([Treatment(start=measurement_times[0], end=measurement_times[-1], id=1)])
        # Plot true M protein curves according to parameters
        plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # sens
        ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b') #, label="True M protein (total)")
        # Plot resistant M protein
        ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r") #, label="True M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k') #, label="True M protein (total)")

    ax1.set_title(title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/L)")
    ax1.set_ylim(bottom=0, top=85) # Not on server
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    plt.close()

def plot_true_mprotein_with_observations_and_treatments_and_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    first_time = min(measurement_times[0], treatment_history[0].start)
    max_time = find_max_time(measurement_times)
    plotting_times = np.linspace(first_time, max_time, int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    # Plot sensitive and resistant
    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="True M protein (resistant)")
    # Plot total M protein
    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', label="True M protein (total)")
    if PLOT_ESTIMATES:
        # Plot estimtated Mprotein line
        estimated_mprotein_values = measure_Mprotein_noiseless(estimated_parameters, plotting_times, treatment_history)
        ax1.plot(plotting_times, estimated_mprotein_values, linestyle='--', linewidth=2, marker='', zorder=3, color='b', label="Estimated M protein")

    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        # Adaptation made to simulation study 2:
        if treat_index>0:
            ax1.axvline(this_treatment.start, color="k", linewidth=1, linestyle="-", label="Start of period of interest")
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start
            if this_treatment.id > maxdrugkey:
                maxdrugkey = this_treatment.id

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary_OSLO[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/L)")
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("Treatment line. max="+str(maxdrugkey))
    ax2.set_yticks(range(maxdrugkey+1))
    ax2.set_yticklabels(range(maxdrugkey+1))
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    if savename != 0:
        plt.savefig(savename,dpi=300)
    else:
        if PLOT_ESTIMATES:
            plt.savefig("./patient_truth_and_observations_with_model_fit"+plot_title+".pdf",dpi=300)
        else:
            plt.savefig("./patient_truth_and_observations"+plot_title+".pdf",dpi=300)
    #plt.show()
    plt.close()

def plot_treatment_region_with_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 0

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    # Plot sensitive and resistant
    if true_parameters.pi_r > 10e-10 and true_parameters.pi_r < 1-10e-10: 
        # Plot resistant
        ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")
    elif true_parameters.pi_r > 1-10e-10:
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='r', label="Estimated M protein (total)")
    else:
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")

    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")
    #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start
            if this_treatment.id > maxdrugkey:
                maxdrugkey = this_treatment.id

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary_OSLO[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/dL)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    ax2.set_ylabel("Treatment id for blue region")
    ax2.set_yticks([maxdrugkey])
    ax2.set_yticklabels([maxdrugkey])
    ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    #plt.show()
    plt.close()


def plot_to_compare_estimated_and_predicted_drug_dynamics(true_parameters, predicted_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot predicted M protein values according to predicted parameters
    plotting_mprotein_values_pred = measure_Mprotein_noiseless(predicted_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters_pred = Parameters((predicted_parameters.Y_0*predicted_parameters.pi_r), 1, predicted_parameters.g_r, predicted_parameters.g_s, predicted_parameters.k_1, predicted_parameters.sigma)
    plotting_resistant_mprotein_values_pred = measure_Mprotein_noiseless(resistant_parameters_pred, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    # Plot sensitive and resistant
    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
    # Plot sensitive and resistant
    ax1.plot(plotting_times, plotting_resistant_mprotein_values_pred, linestyle='--', marker='', zorder=3, color='orange', label="Predicted M protein (resistant)")
    # Plot total M protein
    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")
    # Plot total M protein, predicted
    ax1.plot(plotting_times, plotting_mprotein_values_pred, linestyle='--', marker='', zorder=3, color='b', label="Predicted M protein (total)")

    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")
    [ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start
            if this_treatment.id > maxdrugkey:
                maxdrugkey = this_treatment.id

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary_OSLO[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/dL)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    ax2.set_ylabel("Treatment line. max="+str(maxdrugkey))
    ax2.set_yticks(range(maxdrugkey+1))
    ax2.set_yticklabels(range(maxdrugkey+1))
    ax2.set_ylim(bottom=0, top=maxdrugkey+plotheight/2)
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    #plt.show()
    plt.close()

# Plot posterior confidence intervals 
def plot_posterior_confidence_intervals(training_instance_id, patient, sorted_pred_y_values, parameter_estimates=[], PLOT_POINT_ESTIMATES=False, PLOT_TREATMENTS=False, plot_title="", savename="0", y_resolution=1000, n_chains=4, n_samples=1000):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, y_resolution)
    
    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')

    if PLOT_POINT_ESTIMATES:
        # Plot true M protein values according to parameter estimates
        plotting_mprotein_values = measure_Mprotein_noiseless(parameter_estimates, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameter_estimates.Y_0*parameter_estimates.pi_r), 1, parameter_estimates.g_r, parameter_estimates.g_s, parameter_estimates.k_1, parameter_estimates.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # Plot resistant M protein
        ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")

    # Plot posterior confidence intervals 
    # 95 % empirical confidence interval
    color_array = ["#fbd1b4", "#f89856", "#e36209"] #["#fbd1b4", "#fab858", "#f89856", "#f67c27", "#e36209"] #https://icolorpalette.com/color/rust-orange
    for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
        # Get index to find right value 
        lower_index = int(critical_value*sorted_pred_y_values.shape[0]) #n_chains*n_samples)
        upper_index = int((1-critical_value)*sorted_pred_y_values.shape[0]) #n_chains*n_samples)
        # index at intervals to get 95 % limit value
        lower_limits = sorted_pred_y_values[lower_index,training_instance_id,:]
        upper_limits = sorted_pred_y_values[upper_index,training_instance_id,:]       #color=color_array[index]
        ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.copper(1-critical_value), label='%3.0f %% confidence band on M protein value' % (100*(1-2*critical_value)))

    # Plot M protein observations
    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein") #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    if PLOT_TREATMENTS:
        plotheight = 1
        maxdrugkey = 0
        ax2 = ax1.twinx() 
        for treat_index in range(len(treatment_history)):
            this_treatment = treatment_history[treat_index]
            if this_treatment.id != 0:
                treatment_duration = this_treatment.end - this_treatment.start
                if this_treatment.id > maxdrugkey:
                    maxdrugkey = this_treatment.id
                ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/dL)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    if PLOT_TREATMENTS:
        ax2.set_ylabel("Treatment id for blue region")
        ax2.set_yticks([maxdrugkey])
        ax2.set_yticklabels([maxdrugkey])
        ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    #plt.show()
    plt.close()

def plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=[], PLOT_PARAMETERS=False, PLOT_TREATMENTS=False, plot_title="", savename="0", y_resolution=1000, n_chains=4, n_samples=1000, sorted_resistant_mprotein=[], PLOT_MEASUREMENTS = True, PLOT_RESISTANT=True, IS_TEST_PATIENT=False, CLIP_MPROTEIN_TIME=0, clinic_view=False, p_progression=[], time_scale=1, M_scale=1, plot_pfs=False, plot_KapLam=False, print_p_progression=False):
    Mprotein_values = patient.get_Mprotein_values() * M_scale
    measurement_times = patient.get_measurement_times() * time_scale
    ######
    # Scale back
    patient.Mprotein_values = patient.Mprotein_values * M_scale
    patient.measurement_times = patient.measurement_times * time_scale
    for covariate_name, time_series in patient.longitudinal_data.items():
        patient.longitudinal_times[covariate_name] = patient.longitudinal_times[covariate_name] * time_scale
    ######
    if clinic_view:
        treatment_history = np.array([Treatment(start=1, end=CLIP_MPROTEIN_TIME+6*28, id=1)])
        time_max = CLIP_MPROTEIN_TIME+6*28
    else:
        treatment_history = patient.get_treatment_history()
        time_max = find_max_time(measurement_times)
    time_zero = min(treatment_history[0].start, measurement_times[0])
    plotting_times = np.linspace(time_zero, time_max, y_resolution)
    
    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')

    if IS_TEST_PATIENT:
        ax1.axvline(CLIP_MPROTEIN_TIME, color="k", linewidth=1, linestyle="-", zorder=3.9)

    # Plot posterior confidence intervals for Resistant M protein
    # 95 % empirical confidence interval
    if PLOT_RESISTANT:
        if len(sorted_resistant_mprotein) > 0: 
            for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
                # Get index to find right value 
                lower_index = int(critical_value*sorted_resistant_mprotein.shape[0]) #n_chains*n_samples)
                upper_index = int((1-critical_value)*sorted_resistant_mprotein.shape[0]) #n_chains*n_samples)
                # index at intervals to get 95 % limit value
                lower_limits = sorted_resistant_mprotein[lower_index,:]
                upper_limits = sorted_resistant_mprotein[upper_index,:]
                ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.copper(1-critical_value), label='%3.0f %% CI, resistant M protein' % (100*(1-2*critical_value)), zorder=0+index*0.1)

    # Plot posterior confidence intervals for total M protein
    # 95 % empirical confidence interval
    color_array = ["#fbd1b4", "#f89856", "#e36209"] #["#fbd1b4", "#fab858", "#f89856", "#f67c27", "#e36209"] #https://icolorpalette.com/color/rust-orange
    for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
        # Get index to find right value 
        lower_index = int(critical_value*sorted_local_pred_y_values.shape[0]) #n_chains*n_samples)
        upper_index = int((1-critical_value)*sorted_local_pred_y_values.shape[0]) #n_chains*n_samples)
        # index at intervals to get 95 % limit value
        lower_limits = sorted_local_pred_y_values[lower_index,:]
        upper_limits = sorted_local_pred_y_values[upper_index,:]
        shade_array = [0.7, 0.5, 0.35]
        ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.bone(shade_array[index]), label='%3.0f %% CI, total M protein' % (100*(1-2*critical_value)), zorder=1+index*0.1)

    if PLOT_PARAMETERS:
        # Plot true M protein curves according to parameters
        plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # Plot resistant M protein
        if PLOT_RESISTANT:
            ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color=plt.cm.hot(0.2), label="True M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='cyan', label="True M protein (total)")

    # Plot M protein observations
    if PLOT_MEASUREMENTS == True:
        if IS_TEST_PATIENT:
            ax1.plot(measurement_times[measurement_times<=CLIP_MPROTEIN_TIME], Mprotein_values[measurement_times<=CLIP_MPROTEIN_TIME], linestyle='', marker='x', zorder=4, color='k', label="Observed M protein")
            if not clinic_view:
                # plot the unseen observations
                ax1.plot(measurement_times[measurement_times>CLIP_MPROTEIN_TIME], Mprotein_values[measurement_times>CLIP_MPROTEIN_TIME], linestyle='', markersize=6, markeredgewidth=1.7, marker='x', zorder=3.99, color='k', label="Unobserved M protein")
                ax1.plot(measurement_times[measurement_times>CLIP_MPROTEIN_TIME], Mprotein_values[measurement_times>CLIP_MPROTEIN_TIME], linestyle='', markersize=5.5, markeredgewidth=1, marker='x', zorder=4, color='whitesmoke', label="Unobserved M protein")
        else: 
            ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=4, color='k', label="Observed M protein") #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    if PLOT_TREATMENTS:
        plotheight = 1
        maxdrugkey = 0
        ax2 = ax1.twinx()
        for treat_index in range(len(treatment_history)):
            this_treatment = treatment_history[treat_index]
            if this_treatment.id != 0:
                treatment_duration = this_treatment.end - this_treatment.start
                if this_treatment.id > maxdrugkey:
                    maxdrugkey = this_treatment.id
                ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=0, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    if plot_pfs == True and not clinic_view:
        ax1.axvline(patient.pfs, color="r", linewidth=1, linestyle="--", label="PFS") # This was never scaled
    if plot_KapLam:
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()
        try:
            Kappa_values = patient.longitudinal_data["Kappa Light Chain, Free"]
            Kappa_times = patient.longitudinal_times["Kappa Light Chain, Free"]
        except:
            Kappa_values = np.array([])
            Kappa_times = np.array([])

        try:
            Lambda_values = patient.longitudinal_data["Lambda Light Chain, Free"]
            Lambda_times = patient.longitudinal_times["Lambda Light Chain, Free"]
        except:
            Lambda_values = np.array([])
            Lambda_times = np.array([])
        
        try:
            new_ratio_values = patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free"]
            new_ratio_times = patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free"]
        except:
            new_ratio_values = np.array([])
            new_ratio_times = np.array([])
        
        if clinic_view:
            before_clip_mask = Kappa_times <= CLIP_MPROTEIN_TIME
            ax3.plot(Kappa_times[before_clip_mask], Kappa_values[before_clip_mask], linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(0.9), label="Kappa Light Chain SDTM")
            before_clip_mask = Lambda_times <= CLIP_MPROTEIN_TIME
            ax4.plot(Lambda_times[before_clip_mask], Lambda_values[before_clip_mask], linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(1.0), label="Lambda Light Chain SDTM")
            before_clip_mask = Lambda_times <= CLIP_MPROTEIN_TIME
            ax1.plot(new_ratio_times[before_clip_mask], new_ratio_values[before_clip_mask], linestyle='--', marker='x', zorder=2.1, color=plt.cm.viridis(0.8), label="Kappa/Lambda ratio SDTM")
            before_clip_mask = patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"] <= CLIP_MPROTEIN_TIME
            ax1.plot(patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"][before_clip_mask], patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"][before_clip_mask], linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
            plot_title = plot_title + " : " + patient.name
        else:
            #ax1.plot(measurement_times, patient.KappaLambdaRatio, linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
            ax3.plot(Kappa_times, Kappa_values, linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(0.9), label="Kappa Light Chain SDTM")
            ax4.plot(Lambda_times, Lambda_values, linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(1.0), label="Lambda Light Chain SDTM")
            ax1.plot(new_ratio_times, new_ratio_values, linestyle='--', marker='x', zorder=2.1, color=plt.cm.viridis(0.8), label="Kappa/Lambda ratio SDTM")
            ax1.plot(patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
            plot_title = plot_title + " : " + patient.name

    if print_p_progression:
        plot_title = plot_title + "\nP(progress within 6 cycles) = " + str(p_progression[ii - 137])
    ax1.set_title(plot_title)

    if clinic_view:
        latest_cycle_start = int(np.rint( ((CLIP_MPROTEIN_TIME+6*28) - 1) // 28 + 1 ))
    else:
        latest_cycle_start = int(np.rint( (max(measurement_times) - 1) // 28 + 1 ))
    tick_labels = [1] + [cs for cs in range(6, latest_cycle_start+1, 6)]
    #step_size = 6
    ##while len(tick_labels) < min(9, latest_cycle_start):
    ##    step_size = step_size - 1
    ##    tick_labels = [1] + [cs for cs in range(1 + step_size, latest_cycle_start, step_size)]
    #if len(tick_labels) < 4:
    #    step_size = 1
    #    tick_labels = [1] + [cs for cs in range(1 + step_size, latest_cycle_start, step_size)] + [latest_cycle_start]
    if len(tick_labels) < 2:
        tick_labels = [1, latest_cycle_start]
    tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
    ax1.set_xticks(tick_positions, tick_labels)
    ax1.set_xlabel("Cycles")
    if clinic_view:
        ax1.set_xlim(right=CLIP_MPROTEIN_TIME + 6*28)
    #cycle_data = [(mtime - 1) // 28 + 1 for mtime in measurement_times]
    #num_ticks = min(len(cycle_data), 7)
    #step_size = len(cycle_data) // num_ticks
    #tick_positions = measurement_times[::step_size]
    #tick_labels = cycle_data[::step_size]
    #tick_labels = [int(label) for label in tick_labels]
    
    #ax1.set_xticks(measurement_times, cycle_data)

    #tick_positions = measurement_times[1::28]
    #cycles = [xx for xx in tick_positions]
    #ax1.set_xticks(tick_positions, cycles)
    #ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/L)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    if plot_KapLam:
        ax3.set_ylabel("Kappa Light Chain, Free")
        ax3.set_ylim(bottom=0, top=((1.1*max(Kappa_values)) if len(Kappa_values)>1 else 1))
        ax4.set_ylabel("Lambda Light Chain, Free")
        ax4.set_ylim(bottom=0, top=((1.1*max(Lambda_values)) if len(Lambda_values)>1 else 1))
    #ax1.set_xlim(left=time_zero)
    if PLOT_TREATMENTS:
        ax2.set_ylabel("Treatment id for blue region")
        ax2.set_yticks([maxdrugkey])
        ax2.set_yticklabels([maxdrugkey])
        ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    if plot_KapLam:
        ax3.set_zorder(ax1.get_zorder()+1)
        ax4.set_zorder(ax1.get_zorder()+2)
    #handles, labels = ax1.get_legend_handles_labels()
    #lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    #ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename, dpi=300) #, bbox_extra_artists=(lgd), bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_posterior_traces(idata, SAVEDIR, name, psi_prior, model_name, patientwise=True, net_list=["rho_s", "rho_r", "pi_r"], INFERENCE_MODE="Full"):
    if model_name == "linear":
        print("Plotting posterior/trace plots")
        # Autocorrelation plots: 
        az.plot_autocorr(idata, var_names=["sigma_obs"])
        plt.close()

        az.plot_trace(idata, var_names=('alpha', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma_obs'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_group_parameters.pdf")
        plt.close()

        # Combined means combine the chains into one posterior. Compact means split into different subplots
        az.plot_trace(idata, var_names=('beta_rho_r'), lines=[('beta_rho_r', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_rho_r.pdf")
        #plt.show()
        plt.close()

        # Combined means combine the chains into one posterior. Compact means split into different subplots
        az.plot_trace(idata, var_names=('beta_pi_r'), lines=[('beta_pi_r', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_pi_r.pdf")
        #plt.show()
        plt.close()

        az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-forest_beta_rho_r.pdf")
        #plt.show()
        plt.close()
        az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-forest_beta_pi_r.pdf")
        #plt.show()
        plt.close()
        """
        az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.savefig(SAVEDIR+name+"-forest_pi_r.pdf")
        plt.tight_layout()
        #plt.show()
        plt.close()
        """
    elif model_name == "BNN":
        if "rho_s" in net_list:
            # Plot weights in_1 rho_s
            az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=False, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_s.pdf")
            plt.close()
            # Plot weights in_1 rho_s. Combined means combined chains
            az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=True, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_s_combined.pdf")
            plt.close()
            # Plot weights 2_out rho_s
            if INFERENCE_MODE == "Full":
                az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=False, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_s.pdf")
                plt.close()
                # Plot weights 2_out rho_s
                az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=True, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_s_combined.pdf")
                plt.close()

        if "rho_r" in net_list:
            # Plot weights in_1 rho_r
            az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=False, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_r.pdf")
            plt.close()
            # Plot weights in_1 rho_r
            az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=True, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_r_combined.pdf")
            plt.close()
            if INFERENCE_MODE == "Full":
                # Plot weights 2_out rho_r
                az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=False, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_r.pdf")
                plt.close()
                # Plot weights 2_out rho_r
                az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=True, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_r_combined.pdf")
                plt.close()

        if "pi_r" in net_list:
                # Plot weights in_1 pi_r
                az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=False, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_in_1_pi_r.pdf")
                plt.close()
                # Plot weights in_1 pi_r
                az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=True, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_in_1_pi_r_combined.pdf")
                plt.close()
                if INFERENCE_MODE == "Full":
                    # Plot weights 2_out pi_r
                    az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=False, compact=False)
                    plt.tight_layout()
                    plt.savefig(SAVEDIR+name+"-_wts_out_pi_r.pdf")
                    plt.close()
                    # Plot weights 2_out pi_r
                    az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=True, compact=False)
                    plt.tight_layout()
                    plt.savefig(SAVEDIR+name+"-_wts_out_pi_r_combined.pdf")
                    plt.close()

    elif model_name == "joint_BNN":
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out.pdf")
        plt.close()

        # Combined means combined chains
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_combined.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_combined.pdf")
        plt.close()

    if psi_prior=="lognormal":
        az.plot_trace(idata, var_names=('xi'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_group_parameters_xi.pdf")
        plt.close()
    # Test of exploration 
    az.plot_energy(idata)
    plt.savefig(SAVEDIR+name+"-plot_energy.pdf")
    plt.close()
    # Plot of coefficients
    az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_alpha.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_rho_s.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_rho_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_pi_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["psi"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_psi.pdf")
    plt.close()
    if patientwise:
        az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_individual_parameters.pdf")
        plt.close()

def plot_posterior_CI(args):
    sample_shape, y_resolution, ii, idata, patient_dictionary, SAVEDIR, name, N_rand_obs_pred_train, model_name, parameter_dictionary, PLOT_PARAMETERS, CI_with_obs_noise, PLOT_RESISTANT = args
    if not CI_with_obs_noise:
        N_rand_obs_pred_train = 1
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    np.random.seed(ii) # Seeding the randomness in observation noise sigma

    patient = patient_dictionary[ii]
    measurement_times = patient.get_measurement_times() 
    treatment_history = patient.get_treatment_history()
    first_time = min(measurement_times[0], treatment_history[0].start)
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(first_time, time_max, y_resolution) #int((measurement_times[-1]+1)*10))
    posterior_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
    predicted_y_values = np.empty(shape=(n_chains, n_samples*N_rand_obs_pred_train, y_resolution))
    predicted_y_resistant_values = np.empty_like(predicted_y_values)
    for ch in range(n_chains):
        for sa in range(n_samples):
            this_sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
            this_psi       = np.ravel(idata.posterior['psi'][ch,sa,ii])
            this_pi_r      = np.ravel(idata.posterior['pi_r'][ch,sa,ii])
            this_rho_s     = np.ravel(idata.posterior['rho_s'][ch,sa,ii])
            this_rho_r     = np.ravel(idata.posterior['rho_r'][ch,sa,ii])
            posterior_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=this_pi_r, g_r=this_rho_r, g_s=this_rho_s, k_1=0, sigma=this_sigma_obs)
            these_parameters = posterior_parameters[ch,sa]
            resistant_parameters = Parameters((these_parameters.Y_0*these_parameters.pi_r), 1, these_parameters.g_r, these_parameters.g_s, these_parameters.k_1, these_parameters.sigma)
            # Predicted total and resistant M protein
            predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, plotting_times, treatment_history)
            predicted_y_resistant_values_noiseless = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
            # Add noise and make the resistant part the estimated fraction of the observed value
            if CI_with_obs_noise:
                for rr in range(N_rand_obs_pred_train):
                    noise_array = np.random.normal(0, this_sigma_obs, y_resolution)
                    noisy_observations = predicted_y_values_noiseless + noise_array
                    predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                    predicted_y_resistant_values[ch, N_rand_obs_pred_train*sa + rr] = predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
            else: 
                predicted_y_values[ch, sa] = predicted_y_values_noiseless
                predicted_y_resistant_values[ch, sa] = predicted_y_values[ch, sa] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
    flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*n_samples*N_rand_obs_pred_train,y_resolution))
    sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
    flat_pred_resistant = np.reshape(predicted_y_resistant_values, (n_chains*n_samples*N_rand_obs_pred_train,y_resolution))
    sorted_pred_resistant = np.sort(flat_pred_resistant, axis=0)
    savename = SAVEDIR+"CI_training_id_"+str(ii)+"_"+name+".pdf"
    if PLOT_PARAMETERS and len(parameter_dictionary) > 0:
        parameters_ii = parameter_dictionary[ii]
    else: 
        parameters_ii = []
    plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=parameters_ii, PLOT_PARAMETERS=PLOT_PARAMETERS, PLOT_TREATMENTS=False, plot_title="Posterior CI for training patient "+str(ii), savename=savename, y_resolution=y_resolution, n_chains=n_chains, n_samples=n_samples, sorted_resistant_mprotein=sorted_pred_resistant, PLOT_RESISTANT=PLOT_RESISTANT)
    return 0 # {"posterior_parameters" : posterior_parameters, "predicted_y_values" : predicted_y_values, "predicted_y_resistant_values" : predicted_y_resistant_values}

def plot_predictions(args): # Predicts observations of M protein
    #sample_shape, y_resolution, ii = args
    sample_shape, y_resolution, ii, idata, X_test, patient_dictionary_test, SAVEDIR, name, N_rand_eff_pred, N_rand_obs_pred, model_name, parameter_dictionary, PLOT_PARAMETERS, PLOT_TREATMENTS, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, PLOT_RESISTANT, PLOT_MEASUREMENTS, net_list, INFERENCE_MODE, MAP_weights = args
    if not CI_with_obs_noise:
        N_rand_eff_pred = N_rand_eff_pred * N_rand_obs_pred
        N_rand_obs_pred = 1
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    np.random.seed(ii) # Seeding the randomness in observation noise sigma, in random effects and in psi = yi0 + random(sigma)

    patient = patient_dictionary_test[ii]
    measurement_times = patient.get_measurement_times() 
    treatment_history = patient.get_treatment_history()
    first_time = min(measurement_times[0], treatment_history[0].start)
    max_time = find_max_time(measurement_times)
    plotting_times = np.linspace(first_time, max_time, y_resolution) #int((measurement_times[-1]+1)*10))
    predicted_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
    predicted_y_values = np.empty(shape=(n_chains*N_rand_eff_pred, n_samples*N_rand_obs_pred, y_resolution))
    predicted_y_resistant_values = np.empty_like(predicted_y_values)
    for ch in range(n_chains):
        for sa in range(n_samples):
            sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
            alpha = np.ravel(idata.posterior['alpha'][ch,sa])

            if model_name == "linear": 
                #this_beta_rho_s = np.ravel(idata.posterior['beta_rho_s'][ch,sa])
                this_beta_rho_r = np.ravel(idata.posterior['beta_rho_r'][ch,sa])
                this_beta_pi_r = np.ravel(idata.posterior['beta_pi_r'][ch,sa])
            elif model_name == "BNN": 
                if "rho_s" in net_list:
                    weights_in_rho_s = idata.posterior['weights_in_rho_s'][ch,sa]
                    if INFERENCE_MODE == "Full":
                        weights_out_rho_s = idata.posterior['weights_out_rho_s'][ch,sa]
                    elif INFERENCE_MODE == "Partial": 
                        weights_out_rho_s = MAP_weights["weights_out_rho_s"]
                    bias_in_rho_s = np.ravel(idata.posterior['bias_in_rho_s'][ch,sa])
                    pre_act_1_rho_s = np.dot(X_test.iloc[ii,:], weights_in_rho_s) + bias_in_rho_s
                    act_1_rho_s = np.select([pre_act_1_rho_s > 0, pre_act_1_rho_s <= 0], [pre_act_1_rho_s, pre_act_1_rho_s*0.01], 0)
                    act_out_rho_s = np.dot(act_1_rho_s, weights_out_rho_s)
                else: 
                    act_out_rho_s = 0

                if "rho_r" in net_list:
                    weights_in_rho_r = idata.posterior['weights_in_rho_r'][ch,sa]
                    if INFERENCE_MODE == "Full":
                        weights_out_rho_r = idata.posterior['weights_out_rho_r'][ch,sa]
                    elif INFERENCE_MODE == "Partial": 
                        weights_out_rho_r = MAP_weights["weights_out_rho_r"]
                    bias_in_rho_r = np.ravel(idata.posterior['bias_in_rho_r'][ch,sa])
                    pre_act_1_rho_r = np.dot(X_test.iloc[ii,:], weights_in_rho_r) + bias_in_rho_r
                    act_1_rho_r = np.select([pre_act_1_rho_r > 0, pre_act_1_rho_r <= 0], [pre_act_1_rho_r, pre_act_1_rho_r*0.01], 0)
                    act_out_rho_r = np.dot(act_1_rho_r, weights_out_rho_r)
                else:
                    act_out_rho_r = 0

                if "pi_r" in net_list:
                    weights_in_pi_r = idata.posterior['weights_in_pi_r'][ch,sa]
                    if INFERENCE_MODE == "Full":
                        weights_out_pi_r = idata.posterior['weights_out_pi_r'][ch,sa]
                    elif INFERENCE_MODE == "Partial": 
                        weights_out_pi_r = MAP_weights["weights_out_pi_r"]
                    bias_in_pi_r = np.ravel(idata.posterior['bias_in_pi_r'][ch,sa])
                    pre_act_1_pi_r  = np.dot(X_test.iloc[ii,:], weights_in_pi_r)  + bias_in_pi_r
                    act_1_pi_r =  np.select([pre_act_1_pi_r  > 0, pre_act_1_pi_r  <= 0], [pre_act_1_pi_r,  pre_act_1_pi_r*0.01],  0)
                    act_out_pi_r =  np.dot(act_1_pi_r,  weights_out_pi_r)
                else:
                    act_out_pi_r = 0

            elif model_name == "joint_BNN": 
                # weights 
                weights_in = idata.posterior['weights_in'][ch,sa]
                weights_out = idata.posterior['weights_out'][ch,sa]

                # intercepts
                #sigma_bias_in = idata.posterior['sigma_bias_in'][ch,sa]
                bias_in = np.ravel(idata.posterior['bias_in'][ch,sa])

                pre_act_1 = np.dot(X_test.iloc[ii,:], weights_in) + bias_in

                act_1 = np.select([pre_act_1 > 0, pre_act_1 <= 0], [pre_act_1, pre_act_1*0.01], 0)

                # Output
                act_out = np.dot(act_1, weights_out)
                act_out_rho_s = act_out[0]
                act_out_rho_r = act_out[1]
                act_out_pi_r =  act_out[2]

            # Random effects 
            omega  = np.ravel(idata.posterior['omega'][ch,sa])
            for ee in range(N_rand_eff_pred):
                if model_name == "linear":
                    #if MODEL_RANDOM_EFFECTS: 
                    #predicted_theta_1 = np.random.normal(alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s), omega[0])
                    predicted_theta_1 = np.random.normal(alpha[0], omega[0])
                    predicted_theta_2 = np.random.normal(alpha[1] + np.dot(X_test.iloc[ii,:], this_beta_rho_r), omega[1])
                    predicted_theta_3 = np.random.normal(alpha[2] + np.dot(X_test.iloc[ii,:], this_beta_pi_r), omega[2])
                    #else: 
                    #    predicted_theta_1 = alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s)
                    #    predicted_theta_2 = alpha[1] + np.dot(X_test.iloc[ii,:], this_beta_rho_r)
                    #    predicted_theta_3 = alpha[2] + np.dot(X_test.iloc[ii,:], this_beta_pi_r)
                elif model_name == "BNN" or model_name == "joint_BNN":
                    if MODEL_RANDOM_EFFECTS:
                        predicted_theta_1 = np.random.normal(alpha[0] + act_out_rho_s, omega[0])
                        predicted_theta_2 = np.random.normal(alpha[1] + act_out_rho_r, omega[1])
                        predicted_theta_3 = np.random.normal(alpha[2] + act_out_pi_r, omega[2])
                    else: 
                        predicted_theta_1 = alpha[0] + act_out_rho_s
                        predicted_theta_2 = alpha[1] + act_out_rho_r
                        predicted_theta_3 = alpha[2] + act_out_pi_r

                predicted_rho_s = - np.exp(predicted_theta_1)
                predicted_rho_r = np.exp(predicted_theta_2)
                predicted_pi_r  = 1/(1+np.exp(-predicted_theta_3))

                this_psi = patient.Mprotein_values[0] + np.random.normal(0,sigma_obs)
                predicted_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=predicted_pi_r, g_r=predicted_rho_r, g_s=predicted_rho_s, k_1=0, sigma=sigma_obs)
                these_parameters = predicted_parameters[ch,sa]
                resistant_parameters = Parameters(Y_0=(these_parameters.Y_0*these_parameters.pi_r), pi_r=1, g_r=these_parameters.g_r, g_s=these_parameters.g_s, k_1=these_parameters.k_1, sigma=these_parameters.sigma)
                # Predicted total and resistant M protein
                predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, plotting_times, treatment_history)
                predicted_y_resistant_values_noiseless = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Add noise and make the resistant part the estimated fraction of the observed value
                if CI_with_obs_noise:
                    for rr in range(N_rand_obs_pred):
                        noise_array = np.random.normal(0, sigma_obs, y_resolution)
                        noisy_observations = predicted_y_values_noiseless + noise_array
                        predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                        predicted_y_resistant_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
                else: 
                    predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa] = predicted_y_values_noiseless
                    predicted_y_resistant_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa] = predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
    flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*n_samples*N_rand_eff_pred*N_rand_obs_pred,y_resolution))
    sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
    flat_pred_resistant = np.reshape(predicted_y_resistant_values, (n_chains*n_samples*N_rand_eff_pred*N_rand_obs_pred,y_resolution))
    sorted_pred_resistant = np.sort(flat_pred_resistant, axis=0)
    savename = SAVEDIR+"CI_test_id_"+str(ii)+"_"+name+".pdf"
    if PLOT_PARAMETERS and len(parameter_dictionary) > 0:
        parameters_ii = parameter_dictionary[ii]
    else:
        parameters_ii = []
    plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=parameters_ii, PLOT_PARAMETERS=PLOT_PARAMETERS, plot_title="Posterior predictive CI for test patient "+str(ii), savename=savename, y_resolution=y_resolution, n_chains=n_chains, n_samples=n_samples, sorted_resistant_mprotein=sorted_pred_resistant, PLOT_MEASUREMENTS = PLOT_MEASUREMENTS, PLOT_RESISTANT=PLOT_RESISTANT)
    return 0 # {"posterior_parameters" : posterior_parameters, "predicted_y_values" : predicted_y_values, "predicted_y_resistant_values" : predicted_y_resistant_values}

def pfs_auc(evaluation_time, patient_dictionary_test, N_patients_test, idata, X_test, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, SAVEDIR, name, INFERENCE_MODE, MAP_weights, net_list):
    return # dsicontinued because old relapse definition
    N_patients_test = len(patient_dictionary_test)

    recurrence_or_not = np.zeros(N_patients_test)
    for ii in range(N_patients_test):
        patient = patient_dictionary_test[ii]
        mprot = patient.Mprotein_values
        times = patient.measurement_times
        # Old criteria: Going above the initial M protein level 
        #recurrence_or_not[ii] = int( (mprot[1:][times[1:] < evaluation_time] > mprot[0]).any() )
        # New criteria from https://www.myelomacentral.com/livingwithmm/multiple-myeloma-symptoms-and-diagnosis/glossary-multiple-myeloma-definitions 
        # At least one of the following: 
        # a 25% increase in serum M-component with an absolute increase of at least 0.5 g/dL; 
        # Loop through measurements and find first case 
        previous_Mprotein = mprot[0]
        lowest_Mprotein = mprot[0]
        for tt, _ in enumerate(times): 
            if tt == 0: 
                continue
            if times[tt] > evaluation_time: 
                break
            if mprot[tt] > previous_Mprotein: 
                if mprot[tt] > lowest_Mprotein*1.25 and mprot[tt] - lowest_Mprotein > 0.5:
                    recurrence_or_not[ii] = 1
                    break
            previous_Mprotein = mprot[tt]
            lowest_Mprotein = min(lowest_Mprotein, previous_Mprotein)
        # urine M-component with an absolute increase of at least 200 mg per 24 hours; 
        # <...> empty for now
        # an absolute increase in urine M-protein levels in patients without measurable serum and urine M-protein levels; 
        # <...> empty for now
        # bone marrow plasma cell percentage of at least 10%; 
        # <...> empty for now
        # development of new bone lesions or soft tissue plasmacytomas; 
        # <...> empty for now
        # or development of hypercalcemia.
    #print("Recurrence", recurrence_or_not)
    #print("Proportion with recurrence:", np.mean(recurrence_or_not))

    sample_shape = idata.posterior['psi'].shape # [chain, n_samples, dim]
    N_samples = sample_shape[1]
    # Posterior predictive CI for test data
    if N_samples <= 100:
        N_rand_eff_pred = 10 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 1000:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    else:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 10 # Number of observation noise samples to draw for each parameter sample 

    args = [(sample_shape, ii, idata, X_test, patient_dictionary_test, N_rand_eff_pred, N_rand_obs_pred, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, evaluation_time, INFERENCE_MODE, MAP_weights, net_list) for ii in range(N_patients_test)]

    p_recurrence = np.zeros(N_patients_test) 
    predicted_PFS = np.empty(N_patients_test, dtype=object)
    point_predicted_PFS = np.zeros(N_patients_test) 
    for ii, elem in enumerate(args):
        p_recurrence[ii], predicted_PFS[ii], point_predicted_PFS[ii] = predict_PFS(elem)
    
    #print("recurrence_or_not", recurrence_or_not)
    #print("p_recurrence", p_recurrence)

    #picklefile = open(SAVEDIR+name+'_recurrence_or_not', 'wb')
    #pickle.dump(recurrence_or_not, picklefile)
    #picklefile.close()

    #picklefile = open(SAVEDIR+name+'_p_recurrence', 'wb')
    #pickle.dump(p_recurrence, picklefile)
    #picklefile.close()

    """
    # Commented out only because med-biostat2 does not have sklearn 
    # calculate the fpr and tpr for all thresholds of the classification
    import sklearn.metrics as metrics
    fpr, tpr, threshold = metrics.roc_curve(recurrence_or_not, p_recurrence) #(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    print("fpr:\n", fpr)
    print("tpr:\n", tpr)
    print("threshold:\n", threshold)
    print("roc_auc:", roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(SAVEDIR+"AUC_"+str(N_patients_test)+"_test_patients_"+name+".pdf")
    #plt.show()
    plt.close()
    """
    return recurrence_or_not, p_recurrence, predicted_PFS, point_predicted_PFS

def plot_all_credible_intervals(idata, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name, y_resolution, model_name, parameter_dictionary, PLOT_PARAMETERS, parameter_dictionary_test, PLOT_PARAMETERS_test, PLOT_TREATMENTS, MODEL_RANDOM_EFFECTS, CI_with_obs_noise=True, PARALLELLIZE=True, PLOT_RESISTANT=True, PLOT_MEASUREMENTS_test=False, net_list=["pi", "rho_r", "rho_s"], INFERENCE_MODE="Full", MAP_weights={}):
    sample_shape = idata.posterior['psi'].shape # [chain, N_samples, dim]
    N_chains = sample_shape[0]
    N_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient

    # Posterior CI for train data
    if N_samples <= 10:
        N_rand_obs_pred_train = 10000 # Number of observation noise samples to draw for each parameter sample
    elif N_samples <= 100:
        N_rand_obs_pred_train = 1000 # Number of observation noise samples to draw for each parameter sample
    elif N_samples <= 1000:
        N_rand_obs_pred_train = 100 # Number of observation noise samples to draw for each parameter sample
    else:
        N_rand_obs_pred_train = 10 # Number of observation noise samples to draw for each parameter sample
    print("Plotting posterior credible bands for training cases")
    N_patients = len(patient_dictionary)
    args = [(sample_shape, y_resolution, ii, idata, patient_dictionary, SAVEDIR, name, N_rand_obs_pred_train, model_name, parameter_dictionary, PLOT_PARAMETERS, CI_with_obs_noise, PLOT_RESISTANT) for ii in range(min(N_patients, 20))]
    if PARALLELLIZE:
        if SAVEDIR in ["./plots/Bayesian_estimates_simdata_linearmodel/", "./plots/Bayesian_estimates_simdata_BNN/", "./plots/Bayesian_estimates_simdata_joint_BNN/"]:
            poolworkers = 15
        else:
            poolworkers = 4 
        with Pool(poolworkers) as pool:
            results = pool.map(plot_posterior_CI,args)
    else: 
        for elem in args:
            plot_posterior_CI(elem)
    print("...done.")

    # Posterior predictive CI for test data
    if N_samples <= 10:
        N_rand_eff_pred = 100 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 100:
        N_rand_eff_pred = 10 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 1000:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    else:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 10 # Number of observation noise samples to draw for each parameter sample 
    print("Plotting predictive credible bands for test cases")
    N_patients_test = len(patient_dictionary_test)
    args = [(sample_shape, y_resolution, ii, idata, X_test, patient_dictionary_test, SAVEDIR, name, N_rand_eff_pred, N_rand_obs_pred, model_name, parameter_dictionary_test, PLOT_PARAMETERS_test, PLOT_TREATMENTS, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, PLOT_RESISTANT, PLOT_MEASUREMENTS_test, net_list, INFERENCE_MODE, MAP_weights) for ii in range(N_patients_test)]
    if PARALLELLIZE:
        with Pool(poolworkers) as pool:
            results = pool.map(plot_predictions,args)
    else: 
        for elem in args:
            plot_predictions(elem)
    print("...done.")

def plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r):
    color_array = X["Covariate 2"].to_numpy()
    striplets = [("expected_theta_1", "true_theta_rho_s", "true_rho_s"), ("expected_theta_2", "true_theta_rho_r", "true_rho_r"), ("expected_theta_3", "true_theta_pi_r", "true_pi_r")]
    for ii, triplet in enumerate([(expected_theta_1, true_theta_rho_s, true_rho_s), (expected_theta_2, true_theta_rho_r, true_rho_r), (expected_theta_3, true_theta_pi_r, true_pi_r)]):
        striplet = striplets[ii]
        for jj, elem in enumerate(triplet):
            selem = striplet[jj]
            for covariate in ["Covariate 1", "Covariate 3"]:
                # Covariates 1 and 2
                fig, ax = plt.subplots()
                ax.set_title(selem)
                points = ax.scatter(X[covariate], elem, c=color_array, cmap="plasma")
                ax.set_xlabel(covariate)
                ax.set_ylabel(selem)
                cbar = fig.colorbar(points)
                cbar.set_label('covariate 2', rotation=90)
                plt.savefig(SAVEDIR + '_'.join(["effects",covariate,str(ii),str(jj),selem,name,".pdf"]), dpi=300)
                plt.close()

# Plot pi, rho and alpha against each covariate
def plot_parameters_and_covariates(SAVEDIR, name, X, rho_s, rho_r, pi_r, covariates):
    triplet = [rho_s, rho_r, pi_r]
    striplet = ["rho_s", "rho_r", "pi_r"]
    #color_array = X["Covariate 2"].to_numpy()
    N_COLS = 5
    for jj, elem in enumerate(triplet):
        parameter = striplet[jj]
        # Plot in a grid: 
        fig, axes = plt.subplots(ncols=N_COLS, nrows=max(2,1+int(len(covariates)/N_COLS)), figsize=(15,15))
        plt.suptitle(parameter)
        for ii, covariate in enumerate(covariates):
            points = axes[int(ii/N_COLS), ii % N_COLS].scatter(X[covariate], elem, c="k") #c=color_array, cmap="plasma")
            axes[int(ii/N_COLS), ii % N_COLS].set_xlabel(covariate)
        plt.tight_layout()
        plt.savefig(SAVEDIR + '_'.join(["overview",parameter,name,".pdf"]), dpi=300)
        plt.close()
"""
        for covariate in covariates:
            # Covariates 1 and 2
            fig, ax = plt.subplots()
            ax.set_title(parameter)
            points = ax.scatter(X[covariate], elem, c="k") #c=color_array, cmap="plasma")
            ax.set_ylabel(parameter)
            ax.set_xlabel(covariate)
            plt.savefig(SAVEDIR + '_'.join(["overview",covariate,str(jj),parameter,name,".pdf"]), dpi=300)
            plt.close()
"""

#####################################
# Inference
#####################################
global_sigma = 0.1  # Standard deviation of measurement noise

def least_squares_objective_function_patient_1(array_x, measurement_times, treatment_history, observations):
    Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=array_x[1], g_r=array_x[2], g_s=array_x[3], k_1=0.000, sigma=global_sigma)
    predictions = measure_Mprotein_noiseless(Parameter_object_x, measurement_times, treatment_history)
    sumofsquares = np.sum((observations - predictions)**2)
    return sumofsquares

def least_squares_objective_function(array_x, patient):
    measurement_times = patient.measurement_times
    treatment_history = patient.treatment_history
    observations = patient.Mprotein_values

    if len(array_x) == 4:
        array_x = np.concatenate((array_x, [0]))
    Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=array_x[1], g_r=array_x[2], g_s=array_x[3], k_1=array_x[4], sigma=global_sigma)
    predictions = measure_Mprotein_noiseless(Parameter_object_x, measurement_times, treatment_history)
    sumofsquares = np.sum((observations - predictions)**2)
    return sumofsquares

# This function accepts parameter arrays containing all parameters including sigma, for any model 1 2 or 3
def negative_loglikelihood_any_model(array_x, patient):
    #placeholder_sigma = 1 # Used when I did not believe in the MLE sigma for model selection 
    measurement_times = patient.measurement_times
    treatment_history = patient.treatment_history
    observations = patient.Mprotein_values
    # Model 1: exp rho t            (2+1=3 parameters: Y0, rho, sigma)
    if len(array_x) == 3:
        Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=1, g_r=array_x[1], g_s=0, k_1=0, sigma=array_x[2])
    # Model 2: exp t(alpha - k)     (3+1=4 parameters: Y0, alpha, K, sigma)
    elif len(array_x) == 4:
        Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=0, g_r=0, g_s=array_x[1], k_1=array_x[2], sigma=array_x[3])
    # Model 3: Both together.       (5+1=6 parameters: Y0, pi, rho, alpha, K, sigma)
    elif len(array_x) == 6:
        Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=array_x[1], g_r=array_x[2], g_s=array_x[3], k_1=array_x[4], sigma=array_x[5])
    
    predictions = measure_Mprotein_noiseless(Parameter_object_x, measurement_times, treatment_history)
    sumofsquares = np.sum((observations - predictions)**2)
    negative_loglikelihood = (len(observations)/2)*np.log(2*np.pi*Parameter_object_x.sigma**2) + sumofsquares/(2*Parameter_object_x.sigma**2)
    #negative_loglikelihood = (len(observations)/2)*np.log(2*np.pi*placeholder_sigma**2) + sumofsquares/(2*placeholder_sigma**2)
    return negative_loglikelihood

def infer_parameters_simulated_patient(patient, lb, ub, N_iterations=10000):
    bounds_Y_0 = (lb[0], ub[0])
    bounds_pi_r = (lb[1], ub[1])
    bounds_g_r = (lb[2], ub[2])
    bounds_g_s = (lb[3], ub[3])
    bounds_k_1 = (lb[4], ub[4])
    all_bounds = (bounds_Y_0, bounds_pi_r, bounds_g_r, bounds_g_s, bounds_k_1)
    lowest_f_value = np.inf
    best_x = np.array([0,0,0,0,0])
    random_samples = np.random.uniform(0,1,len(ub))
    x0 = lb + np.multiply(random_samples, (ub-lb))
    best_optim = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(patient), bounds=all_bounds, options={'disp':False})
    for iteration in range(N_iterations):
        random_samples = np.random.uniform(0,1,len(ub))
        x0 = lb + np.multiply(random_samples, (ub-lb))
        optimization_result = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(patient), bounds=all_bounds, options={'disp':False})
        if optimization_result.fun < lowest_f_value:
            lowest_f_value = optimization_result.fun
            best_x = optimization_result.x
            best_optim = optimization_result
    return best_optim

# https://stackoverflow.com/questions/12874756/parallel-optimizations-in-scipy
def get_optimization_result(args):
    x0, patient, all_bounds = args
    res = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(patient), bounds=all_bounds, options={'disp':False})
    return res

def estimate_drug_response_parameters(patient, lb, ub, N_iterations=10000, PARALLELLIZE=True):
    global_sigma = 1
    # If the bounds do not include bounds for k_1, we set it to zero before and after optimization
    #bounds_Y_0 = (lb[0], ub[0])
    #bounds_pi_r = (lb[1], ub[1])
    #bounds_g_r = (lb[2], ub[2])
    #bounds_g_s = (lb[3], ub[3])
    #bounds_k_1 = (lb[4], ub[4])
    #all_bounds = (bounds_Y_0, bounds_pi_r, bounds_g_r, bounds_g_s) #, bounds_k_1)
    all_bounds = tuple([(lb[ii], ub[ii]) for ii in range(len(ub))])
    # This block and below keeps the unparallell way
    #lowest_f_value = np.inf
    #random_samples = np.random.uniform(0,1,len(ub))
    #x0 = lb + np.multiply(random_samples, (ub-lb))

    all_random_samples = np.random.uniform(0,1,(N_iterations, len(ub)))
    x0_array = lb + np.multiply(all_random_samples, (ub-lb))

    args = [(x0_array[i],patient,all_bounds) for i in range(len(x0_array))]
    if PARALLELLIZE:
        with Pool(15) as pool:
            optim_results = pool.map(get_optimization_result,args)
    else: 
        optim_results = []
        for elem in args:
            optim_results.append(get_optimization_result(elem))    
    #min_f = min(optim_results, key=lambda x: x.fun)
    fun_value_list = [elem.fun for elem in optim_results]
    min_f_index = fun_value_list.index(min(fun_value_list))
    x_value_list = [elem.x for elem in optim_results]
    best_x = x_value_list[min_f_index]

    #for iteration in range(N_iterations):
    #    #random_samples = np.random.uniform(0,1,len(ub))
    #    #x0 = lb + np.multiply(random_samples, (ub-lb))
    #    optimization_result = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(patient), bounds=all_bounds, options={'disp':False})
    #    if optimization_result.fun < lowest_f_value:
    #        lowest_f_value = optimization_result.fun
    #        best_optim = optimization_result
    #best_x = best_optim.x
    if len(ub) == 5: # k_1 included in parameters, rdug holiday included in interval
        return Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=global_sigma)
    elif len(ub) == 4: # k_1 included in parameters, rdug holiday included in interval
        return Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=0, sigma=global_sigma)

def get_optimization_result_any_model(args):
    x0, patient, all_bounds = args
    res = optimize.minimize(fun=negative_loglikelihood_any_model, x0=x0, args=(patient), bounds=all_bounds, options={'disp':False}, method='SLSQP') # L-BFGS-B chosen automatically with bounds
    return res

def estimate_drug_response_parameters_any_model(patient, lb, ub, N_iterations=10000, PARALLELLIZE=True):
    # This function accepts bounds and parameters of any model. THe length determines the model. 
    all_bounds = tuple([(lb[ii], ub[ii]) for ii in range(len(ub))])
    all_random_samples = np.random.uniform(0,1,(N_iterations, len(ub)))
    x0_array = lb + np.multiply(all_random_samples, (ub-lb))

    args = [(x0_array[i],patient,all_bounds) for i in range(len(x0_array))]
    if PARALLELLIZE:
        with Pool(15) as pool:
            optim_results = pool.map(get_optimization_result_any_model,args)
    else: 
        optim_results = []
        for elem in args:
            optim_results.append(get_optimization_result_any_model(elem))
    fun_value_list = [elem.fun for elem in optim_results]
    min_f_index = fun_value_list.index(min(fun_value_list))
    x_value_list = [elem.x for elem in optim_results]
    best_x = x_value_list[min_f_index]

    # Model 1: exp rho t            (2+1=3 parameters: Y0, rho, sigma)
    if len(lb) == 3:
        Parameter_object_x = Parameters(Y_0=best_x[0], pi_r=1, g_r=best_x[1], g_s=GROWTH_LB, k_1=0, sigma=best_x[2])
    # Model 2: exp t(alpha - k)     (3+1=4 parameters: Y0, alpha, K, sigma)
    elif len(best_x) == 4:
        Parameter_object_x = Parameters(Y_0=best_x[0], pi_r=0, g_r=GROWTH_LB, g_s=best_x[1], k_1=best_x[2], sigma=best_x[3])
    # Model 3: Both together.       (5+1=6 parameters: Y0, pi, rho, alpha, K, sigma)
    elif len(best_x) == 6:
        Parameter_object_x = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=best_x[5])
    return Parameter_object_x

def get_binary_outcome(period_start, patient, this_estimate, days_for_consideration=182):
    # If projected/observed M protein value goes above M protein value at treatment start within X days, then outcome = 1
    # NB! To predict chances under received treatment, we must encode future treatment precisely in covariates to predict effect of future treatment.  This means encode treatment+drug holiday in X.
    # What we do now is to predict outcome under continuous administration of treatment, for the time interval where we predict response.  

    # Using estimated Y_0 as M protein value at treatment start
    initial_Mprotein_value = this_estimate.Y_0
    
    future_starts = np.array([elem.start for elem in patient.treatment_history])
    #future_treatments = patient.treatment_history[future_starts >= period_start]
    #future_treatments = np.array([Treatment(elem.start - period_start, elem.end - period_start, elem.id) for elem in future_treatments])
    first_future_treatment = patient.treatment_history[future_starts >= period_start][0]
    future_treatments = np.array([Treatment(period_start, period_start + days_for_consideration, first_future_treatment.id)])

    # Using predicted Mprotein value to check for increase
    predicted_Mprotein = measure_Mprotein_noiseless(this_estimate, np.array([period_start+days_for_consideration]), future_treatments)
    if predicted_Mprotein[0] > initial_Mprotein_value:
        return int(1)
    # Return nan if the future history is shorter than the period we consider
    elif max(np.array([elem.end for elem in future_treatments]) < days_for_consideration):
        return np.nan
    else: # If we did not observe relapse and the observed period is long enough, report 0
        return 0
    ## Using observed values to check for increase
    #future_times = patient.measurement_times[patient.measurement_times > period_start]
    #future_Mprotein = patient.Mprotein_values[patient.measurement_times > period_start]
    #relevant_times = future_times[future_times <= (period_start+days_for_consideration)]
    #relevant_Mprotein = future_Mprotein[future_times <= (period_start+days_for_consideration)]
    #
    #if len(relevant_Mprotein[relevant_Mprotein > initial_Mprotein_value]) > 0:
    #    binary_outcome = 1
    #elif len(relevant_Mprotein[relevant_Mprotein <= initial_Mprotein_value]) > 0:
    #    binary_outcome = 0
    #else: # if the list has no length, i.e. we have no measurements of it
    #    binary_outcome = np.nan
    #return binary_outcome

def estimate_individual_parameters(patient_dictionary, N_iter=10000, plotting=True, SAVEDIR="./", name="_"):
    negative_loglikelihoods = []
    Y_parameters = []
    rho_s = []
    rho_r = []
    pi_r = []
    N_iter = N_iter # separate minimzations of the least squares when fitting parameters
    #def estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3):
    for _, this_patient in patient_dictionary.items():
        # Estimate parameters for model 3
        # Model 3: Both together.       (5+1=6 parameters: Y0, pi, rho, alpha, K, sigma)
        this_estimate = estimate_drug_response_parameters_any_model(this_patient, lb_3, ub_3, N_iterations=N_iter) #, sigma_noise_std=1
        array_x = np.array([this_estimate.Y_0, this_estimate.pi_r, this_estimate.g_r, this_estimate.g_s, this_estimate.k_1, this_estimate.sigma])
        predictions = measure_Mprotein_noiseless(this_estimate, this_patient.measurement_times, this_patient.treatment_history)
        sumofsquares_model_3 = np.sum((this_patient.Mprotein_values - predictions)**2)
        sample_variance_unadjusted_model_3 = sumofsquares_model_3/len(this_patient.measurement_times)
        negative_loglikelihoods.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_3))
        #print(negative_loglikelihoods[training_instance_id])
        
        Y_parameters.append(this_estimate) # training_instance_id is position in Y_parameters
        rho_s.append(this_estimate.g_s - this_estimate.k_1)
        rho_r.append(this_estimate.g_r)
        pi_r.append(this_estimate.pi_r)

        # Plotting treatment region with estimate (using model 3 estimate)
        #for index, param_set in enumerate(patient.parameter_estimates):
        estimated_parameters = this_estimate # patient.parameter_estimates[index]
        #savename = "./COMMPASS_estimate_plots/treatment_id_"+str(treatment_id_of_interest)+"/Treatment_"+str(treatment_id_of_interest)+"_"+patient.name+"_at_time"+str(period_start)+"Y_0="+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sigma="+str(estimated_parameters.sigma)+".png"
        if plotting: 
            savename = SAVEDIR+name+"_".join(["patient"+this_patient.name,"Y_0="])+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sigma="+str(estimated_parameters.sigma)+".png"
            plot_title = this_patient.name
            plot_treatment_region_with_estimate(estimated_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=plot_title, savename=savename)
    return rho_s, rho_r, pi_r

#####################################
# Posterior evaluation
#####################################
# Convergence checks
def quasi_geweke_test(idata, model_name, first=0.1, last=0.5, intervals=20):
    if first+last > 1:
        print("Overriding input since first+last>1. New first, last = 0.1, 0.5")
        first, last = 0.1, 0.5
    print("Running Geweke test...")
    convergence_flag = True
    if model_name == "linear":
        # 'beta_rho_s'
        var_names = ['alpha', 'beta_rho_r', 'beta_pi_r', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    elif model_name == "BNN":
        var_names = ['alpha', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    else:
        var_names = ['alpha', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    for var_name in var_names:
        sample_shape = idata.posterior[var_name].shape
        n_chains = sample_shape[0]
        n_samples = sample_shape[1]
        var_dims = sample_shape[2]
        for chain in range(n_chains):
            for dim in range(var_dims):
                all_samples = np.ravel(idata.posterior[var_name][chain,:,dim])
                first_part = all_samples[0:int(n_samples*first)]
                last_part = all_samples[n_samples-int(n_samples*last):n_samples]
                z_score = (np.mean(first_part)-np.mean(last_part)) / np.sqrt(np.var(first_part)+np.var(last_part))
                if abs(z_score) >= 1.960:
                    convergence_flag = False
                    #print("Seems like chain",chain,"has not converged in",var_name,"dimension",dim,": z_score is",z_score)
    for var_name in ['sigma_obs']:
        all_samples = np.ravel(idata.posterior[var_name])
        n_samples = len(all_samples)
        first_part = all_samples[0:int(n_samples*first)]
        last_part = all_samples[n_samples-int(n_samples*last):n_samples]
        z_score = (np.mean(first_part)-np.mean(last_part)) / np.sqrt(np.var(first_part)+np.var(last_part))
        if abs(z_score) >= 1.960:
            convergence_flag = False
            print("Seems like chain",chain,"has not converged in",var_name,"dimension",dim,": z_score is",z_score)
    if convergence_flag:
        print("All chains seem to have converged.")
    else:
        print("Seems like some chains did not converge.")
    return 0

################################
# Data simulation
################################

# Function to get expected theta from X
def get_expected_theta_from_X_one_interaction(X): # One interaction: In rho_s only
    # These are the true parameters for a patient with all covariates equal to 0:
    N_patients_local, P = X.shape
    rho_s_population = -0.005
    rho_r_population = 0.001
    pi_r_population = 0.3
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    true_beta_rho_s[0] = 0.8
    true_beta_rho_s[1] = 0
    interaction_beta_x1_x2_rho_s = -1
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 1.0
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.0
    true_beta_pi_r[1] = 1.1

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s) + np.ravel(interaction_beta_x1_x2_rho_s*X["Covariate 1"]*(X["Covariate 2"].T)), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients_local,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3

def get_expected_theta_from_X_2_0(X, simulate_rho_r_dependancy_on_rho_s=False, coef_rho_s_rho_r=0): # pi and rho_s
    # No interactions between covariates. Because we only compare the covariate-less with the linear model, not the BNN or lin+interactions model
    # Effect of rho_s on rho_r: Yes, because we observe it in the real data
    # These parameters have covariates with an effect: 
    # rho_s: No. Because not important to predict. We only predict relapse. 
    # rho_r: Yes. From covariates 0, 1 and 2
    # pi_r: Yes. From covariates 0 and 2

    N_patients_local, P = X.shape
    # These are the true parameters for a patient with all covariates equal to 0:
    rho_s_population = -0.015
    rho_r_population = 0.005
    pi_r_population = 0.01
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    #true_beta_rho_s = np.zeros(P)
    #true_beta_rho_s[0] = 0
    #true_beta_rho_s[1] = 0
    #true_beta_rho_s[2] = 0

    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 0.6
    true_beta_rho_r[2] = 0.2

    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.3
    true_beta_pi_r[1] = 0.4
    true_beta_pi_r[2] = 0.5

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, np.zeros(P)), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients_local,1))
    if simulate_rho_r_dependancy_on_rho_s:
        expected_theta_2 = expected_theta_2 + coef_rho_s_rho_r*expected_theta_1
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3

# Until Oct 10 2023
def get_expected_theta_from_X_2(X): # pi and rho_s
    N_patients_local, P = X.shape
    # These are the true parameters for a patient with all covariates equal to 0:
    rho_s_population = -0.005
    rho_r_population = 0.001
    pi_r_population = 0.3
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    true_beta_rho_s[0] = 0.8
    true_beta_rho_s[1] = 0
    true_beta_rho_s[2] = 0.4
    #true_beta_rho_s[3] = 0.3
    #true_beta_rho_s[4] = 0.2
    interaction_beta_x1_x2_rho_s = -1
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 1.0
    true_beta_rho_r[2] = 0.4
    #true_beta_rho_r[3] = 0.2
    #true_beta_rho_r[4] = 0.1
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.0
    true_beta_pi_r[1] = 1.1
    true_beta_pi_r[2] = 0.1
    #true_beta_pi_r[3] = 0.3
    #true_beta_pi_r[4] = 0.4
    interaction_beta_x2_x3_pi_r = -0.6

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s) + np.ravel(interaction_beta_x1_x2_rho_s*X["Covariate 1"]*(X["Covariate 2"].T)), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients_local,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r)  + np.ravel(interaction_beta_x2_x3_pi_r*X["Covariate 2"]*(X["Covariate 3"].T)), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3

def get_expected_theta_from_X_3_pi_rho(X): # Pi and rho_r
    N_patients_local, P = X.shape
    # These are the true parameters for a patient with all covariates equal to 0:
    rho_s_population = -0.005
    rho_r_population = 0.001
    pi_r_population = 0.3
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    #true_beta_rho_s[0] = 0.8
    #true_beta_rho_s[1] = 0
    #true_beta_rho_s[2] = 0.4
    #true_beta_rho_s[3] = 0.3
    #true_beta_rho_s[4] = 0.2
    #interaction_beta_x1_x2_rho_r = -1
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 1.0
    true_beta_rho_r[2] = 0.4
    #true_beta_rho_r[3] = 0.2
    #true_beta_rho_r[4] = 0.1
    interaction_beta_x1_x2_rho_r = -1
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.0
    true_beta_pi_r[1] = 1.1
    true_beta_pi_r[2] = 0.1
    #true_beta_pi_r[3] = 0.3
    #true_beta_pi_r[4] = 0.4
    interaction_beta_x2_x3_pi_r = 1

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r) + np.ravel(interaction_beta_x1_x2_rho_r*X["Covariate 1"]*(X["Covariate 2"].T)), (N_patients_local,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r)  + np.ravel(interaction_beta_x2_x3_pi_r*X["Covariate 2"]*(X["Covariate 3"].T)), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3

def get_sample_weights(args):
    sample_shape, ii, idata = args
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    pi_r_values      = [np.ravel(idata.posterior['pi_r'][ch,sa,ii]) for ch in range(n_chains) for sa in range(n_samples)]
    rho_s_values     = [np.ravel(idata.posterior['rho_s'][ch,sa,ii]) for ch in range(n_chains) for sa in range(n_samples)]
    rho_r_values     = [np.ravel(idata.posterior['rho_r'][ch,sa,ii]) for ch in range(n_chains) for sa in range(n_samples)]
    mean_rho_s = np.mean(rho_s_values)
    sample_weights_rho_s = np.var(rho_s_values)
    mean_rho_r = np.mean(rho_r_values)
    sample_weights_rho_r = np.var(rho_r_values)
    mean_pi_r = np.mean(pi_r_values)
    sample_weights_pi_r = np.var(pi_r_values)
    return mean_rho_s, sample_weights_rho_s, mean_rho_r, sample_weights_rho_r,  mean_pi_r, sample_weights_pi_r

# This is where we make sure the ii matches between X_train and train_dict: 
def get_ii_indexed_subset_dict(raw_subset_df, full_dict): # Requires raw_subset_df with reset indices
    new_dict= {}
    for ii, row in raw_subset_df.iterrows():
        patient_name = row["USUBJID"]
        new_dict[ii] = full_dict[patient_name]
    return new_dict

def plot_fit_and_predictions(idata, patient_dictionary_full, N_patients_train, SAVEDIR, name, y_resolution, CLIP_MPROTEIN_TIME, CI_with_obs_noise=False, PLOT_RESISTANT=True, PLOT_TRAIN=True, clinic_view=False, p_progression=[], time_scale=1, M_scale=1):
    #######
    # Scale time back
    CLIP_MPROTEIN_TIME = CLIP_MPROTEIN_TIME * time_scale
    #######
    sample_shape = idata.posterior['psi'].shape # [chain, N_samples, dim]
    n_chains = sample_shape[0]
    N_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    # Posterior CI for train data
    if CI_with_obs_noise:
        if N_samples <= 10:
            N_rand_obs_pred_train = 10000 # Number of observation noise samples to draw for each parameter sample
        elif N_samples <= 100:
            N_rand_obs_pred_train = 1000 # Number of observation noise samples to draw for each parameter sample
        elif N_samples <= 1000:
            N_rand_obs_pred_train = 100 # Number of observation noise samples to draw for each parameter sample
        else:
            N_rand_obs_pred_train = 10 # Number of observation noise samples to draw for each parameter sample
    else:
        N_rand_obs_pred_train = 1
    print("Plotting posterior credible bands for all cases")
    for ii, patient in patient_dictionary_full.items():
        measurement_times = patient.get_measurement_times() * time_scale
        if ii < N_patients_train and not PLOT_TRAIN:
            continue
        np.random.seed(ii) # Seeding the randomness in observation noise sigma
        if clinic_view:
            time_max = CLIP_MPROTEIN_TIME + 6*28
            treatment_history = np.array([Treatment(start=1, end=CLIP_MPROTEIN_TIME+6*28, id=1)])
        else:
            time_max = find_max_time(measurement_times)
            treatment_history = patient.get_treatment_history()
        first_time = min(measurement_times[0], treatment_history[0].start)
        plotting_times = np.linspace(first_time, time_max, y_resolution) #int((measurement_times[-1]+1)*10))
        posterior_parameters = np.empty(shape=(n_chains, N_samples), dtype=object)
        predicted_y_values = np.empty(shape=(n_chains, N_samples*N_rand_obs_pred_train, y_resolution))
        predicted_y_resistant_values = np.empty_like(predicted_y_values)
        for ch in range(n_chains):
            for sa in range(N_samples):
                this_sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
                this_psi       = np.ravel(idata.posterior['psi'][ch,sa,ii]) * M_scale
                this_pi_r      = np.ravel(idata.posterior['pi_r'][ch,sa,ii])
                this_rho_s     = np.ravel(idata.posterior['rho_s'][ch,sa,ii]) / time_scale
                this_rho_r     = np.ravel(idata.posterior['rho_r'][ch,sa,ii]) / time_scale
                posterior_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=this_pi_r, g_r=this_rho_r, g_s=this_rho_s, k_1=0, sigma=this_sigma_obs)
                these_parameters = posterior_parameters[ch,sa]
                resistant_parameters = Parameters((these_parameters.Y_0*these_parameters.pi_r), 1, these_parameters.g_r, these_parameters.g_s, these_parameters.k_1, these_parameters.sigma)
                # Predicted total and resistant M protein
                predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, plotting_times, treatment_history)
                predicted_y_resistant_values_noiseless = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Add noise and make the resistant part the estimated fraction of the observed value
                if CI_with_obs_noise:
                    for rr in range(N_rand_obs_pred_train):
                        noise_array = np.random.normal(0, this_sigma_obs, y_resolution)
                        noisy_observations = predicted_y_values_noiseless + noise_array
                        predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                        predicted_y_resistant_values[ch, N_rand_obs_pred_train*sa + rr] = predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
                else: 
                    predicted_y_values[ch, sa] = predicted_y_values_noiseless
                    predicted_y_resistant_values[ch, sa] = predicted_y_values[ch, sa] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
        flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*N_samples*N_rand_obs_pred_train,y_resolution))
        sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
        flat_pred_resistant = np.reshape(predicted_y_resistant_values, (n_chains*N_samples*N_rand_obs_pred_train,y_resolution))
        sorted_pred_resistant = np.sort(flat_pred_resistant, axis=0)
        if ii < N_patients_train:
            IS_TEST_PATIENT = False
            if clinic_view:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_train_" + patient.name + "_clinic_view.pdf"
            else:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_train_" + patient.name + ".pdf"
        else:
            IS_TEST_PATIENT = True
            if clinic_view:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_test_" + patient.name + "_clinic_view.pdf"
            else:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_test_" + patient.name + ".pdf"
        plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=[], PLOT_PARAMETERS=False, PLOT_TREATMENTS=False, plot_title="Posterior CI for patient "+str(ii), savename=savename, y_resolution=y_resolution, n_chains=n_chains, n_samples=N_samples, sorted_resistant_mprotein=sorted_pred_resistant, PLOT_RESISTANT=PLOT_RESISTANT, IS_TEST_PATIENT=IS_TEST_PATIENT, CLIP_MPROTEIN_TIME=CLIP_MPROTEIN_TIME, clinic_view=clinic_view, p_progression=p_progression, time_scale=time_scale, M_scale=M_scale)
    print("...done.")

def predict_PFS_velocity_model(patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon, time_scale=1, M_scale=1):
    #######
    # Scale time back
    CLIP_MPROTEIN_TIME = CLIP_MPROTEIN_TIME * time_scale
    end_of_prediction_horizon = end_of_prediction_horizon * time_scale
    #######
    N_patients_test = len(patient_dictionary_full) - N_patients_train
    p_progression = []
    ii_test = 0
    for ii, patient in patient_dictionary_full.items():
        measurement_times = patient.get_measurement_times() * time_scale
        Mprotein_values = patient.Mprotein_values * M_scale
        if ii >= N_patients_train:
            # Find the lowest limit to qualify as progression
            lowest_Mprotein = min(Mprotein_values[measurement_times <= CLIP_MPROTEIN_TIME])
            prev_mprot = Mprotein_values[measurement_times <= CLIP_MPROTEIN_TIME]
            prev_times = measurement_times[measurement_times <= CLIP_MPROTEIN_TIME]
            pfs_threshold = max(1.25*lowest_Mprotein, lowest_Mprotein + 0.5)
            #print("lowest_Mprotein", lowest_Mprotein)
            #print("prev_mprot", prev_mprot)
            #print("prev_times", prev_times)
            #print("pfs_threshold", pfs_threshold)
            times = measurement_times[measurement_times > CLIP_MPROTEIN_TIME]
            if len(prev_mprot) < 1: # no measurements before
                p_progression.append(-3)
                ii_test = ii_test + 1
            elif len(times) < 1 : # no measurements after
                p_progression.append(-2)
                ii_test = ii_test + 1
            else: 
                # Each measurement can have some observation noise with std 0.25
                N_predictions = 1000
                m_std = 0.5
                np.random.seed(0)
                if len(prev_mprot) < 2: # one measurement only
                    # Each measurement can have some observation noise with std 0.25
                    noise_only_one = np.random.normal(loc=0, scale=m_std, size=N_predictions)
                    m_fonly_one = prev_mprot[-1] + noise_only_one
                    #print("m_velocity", m_velocity)
                    predicted_mprot_endtime = np.array([max(0, value) for value in m_fonly_one]) # 0 threshold
                    p_progression_this_patient = np.sum(predicted_mprot_endtime >= pfs_threshold) / N_predictions
                    p_progression.append(p_progression_this_patient)
                else: # at least two measurements and we can get a velocity
                    noise_first = np.random.normal(loc=0, scale=m_std, size=N_predictions)
                    noise_second = np.random.normal(loc=0, scale=m_std, size=N_predictions)
                    m_first = prev_mprot[-2] + noise_first
                    m_second = prev_mprot[-1] + noise_second
                    m_velocity = (m_second - m_first) / (prev_times[-1] - prev_times[-2])
                    #print("m_velocity", m_velocity)
                    predicted_mprot_endtime = (m_second) + m_velocity * (end_of_prediction_horizon - prev_times[-1])
                    predicted_mprot_endtime = np.array([max(0, value) for value in predicted_mprot_endtime]) # 0 threshold
                    p_progression_this_patient = np.sum(predicted_mprot_endtime >= pfs_threshold) / N_predictions
                    p_progression.append(p_progression_this_patient)
                    #print("Patient ii", ii, "p_progression_this_patient:", p_progression_this_patient)
                ii_test = ii_test + 1
    assert int(ii_test) == N_patients_test
    #print(p_progression)
    return p_progression

def predict_PFS_new(idata, patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon, CI_with_obs_noise=False, y_resolution=100, time_scale=1, M_scale=1):
    #######
    # Scale time back
    CLIP_MPROTEIN_TIME = CLIP_MPROTEIN_TIME * time_scale
    end_of_prediction_horizon = end_of_prediction_horizon * time_scale
    #######
    sample_shape = idata.posterior['psi'].shape # [chain, n_samples, dim]
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    # Posterior CI for train data
    if CI_with_obs_noise:
        if n_samples <= 10:
            N_rand_obs_pred_train = 10000 # Number of observation noise samples to draw for each parameter sample
        elif n_samples <= 100:
            N_rand_obs_pred_train = 1000 # Number of observation noise samples to draw for each parameter sample
        elif n_samples <= 1000:
            N_rand_obs_pred_train = 100 # Number of observation noise samples to draw for each parameter sample
        else:
            N_rand_obs_pred_train = 10 # Number of observation noise samples to draw for each parameter sample
    else:
        N_rand_obs_pred_train = 1
    N_patients_test = len(patient_dictionary_full) - N_patients_train
    p_progression = [] #np.repeat(-1, repeats=N_patients_test)
    ii_test = 0
    for ii, patient in patient_dictionary_full.items():
        ######
        measurement_times = patient.get_measurement_times() * time_scale
        Mprotein_values = patient.Mprotein_values * M_scale
        if ii >= N_patients_train:
            if ii == N_patients_train:
                print("In predict_PFS_new, finding p(progression) for ii =", ii, "and the rest...")
            np.random.seed(ii) # Seeding the randomness in observation noise sigma
            # Find the lowest limit to qualify as progression
            lowest_Mprotein = min(Mprotein_values[measurement_times <= CLIP_MPROTEIN_TIME])
            pfs_threshold = max(1.25*lowest_Mprotein, lowest_Mprotein + 0.5)
            # Removed this. This is just to save time. We still want the estimate for plots even though we cannot compare with the truth
            #times = measurement_times[measurement_times > CLIP_MPROTEIN_TIME]
            #if len(times) < 1:
            #    p_progression.append(-2)
            #    ii_test = ii_test + 1
            #else: 
            pat_hist = patient.treatment_history[0]
            treatment_history = [Treatment(pat_hist.start, end_of_prediction_horizon, pat_hist.id)]
            eval_times = np.linspace(CLIP_MPROTEIN_TIME, end_of_prediction_horizon, y_resolution) #find_max_time(times), y_resolution)
            posterior_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
            predicted_y_values = np.empty(shape=(n_chains, n_samples*N_rand_obs_pred_train, y_resolution))
            for ch in range(n_chains):
                for sa in range(n_samples):
                    this_sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa]) * M_scale
                    this_psi       = np.ravel(idata.posterior['psi'][ch,sa,ii]) * M_scale
                    this_pi_r      = np.ravel(idata.posterior['pi_r'][ch,sa,ii])
                    this_rho_s     = np.ravel(idata.posterior['rho_s'][ch,sa,ii]) / time_scale
                    this_rho_r     = np.ravel(idata.posterior['rho_r'][ch,sa,ii]) / time_scale
                    posterior_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=this_pi_r, g_r=this_rho_r, g_s=this_rho_s, k_1=0, sigma=this_sigma_obs)
                    these_parameters = posterior_parameters[ch,sa]
                    predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, eval_times, treatment_history)
                    #print("Y_0", this_psi, "pi_r",this_pi_r, "g_r",this_rho_r, "g_s",this_rho_s)
                    #print("predicted_y_values_noiseless", predicted_y_values_noiseless)
                    #print(np.sum(predicted_y_values_noiseless >= pfs_threshold))
                    # Add noise and make the resistant part the estimated fraction of the observed value
                    if CI_with_obs_noise:
                        for rr in range(N_rand_obs_pred_train):
                            noise_array = np.random.normal(0, this_sigma_obs, y_resolution)
                            noisy_observations = predicted_y_values_noiseless + noise_array
                            predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                    else: 
                        predicted_y_values[ch, sa] = predicted_y_values_noiseless
            # For each time point after clip, p(progression) is the percentage of trajectories above this point
            #print("sum(predicted_y_values < 0.0001)", np.sum(predicted_y_values < 0.0001))
            N_predictions = n_chains*n_samples*N_rand_obs_pred_train
            flat_pred_y_values = np.reshape(predicted_y_values, (N_predictions, y_resolution))
            sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
            p_progression_each_time = [np.sum(sorted_local_pred_y_values[:,tt] >= pfs_threshold)/N_predictions for tt in range(y_resolution)]
            p_progression.append(np.max(p_progression_each_time))
            ii_test = ii_test + 1
    assert int(ii_test) == N_patients_test
    return p_progression

def get_true_pfs_new(patient_dictionary_test, time_scale=1, M_scale=1): # Regardless of CLIP_MPROTEIN_TIME
    N_patients_test = len(patient_dictionary_test)
    true_pfs = np.repeat(-1, repeats=N_patients_test)
    for ii, (patname, patient) in enumerate(patient_dictionary_test.items()):
        mprot = patient.Mprotein_values * M_scale
        times = patient.measurement_times * time_scale
        # Loop through measurements and find first case
        lowest_Mprotein = mprot[0]
        consecutive_above = 0
        for tt, time in enumerate(times):
            if tt == 0:
                continue
            # New criteria for pfs from protocol: 
            # 25% increase in serum M-component with an absolute increase of at least 0.5 g/dL in two consecutive measurements.
            if mprot[tt] > lowest_Mprotein*1.25 and mprot[tt] - lowest_Mprotein > 0.5:
                consecutive_above = consecutive_above + 1
                if consecutive_above == 2:
                    true_pfs[ii] = time
                    break
            else:
                consecutive_above = 0
            lowest_Mprotein = min(lowest_Mprotein, mprot[tt])
        # urine M-component with an absolute increase of at least 200 mg per 24 hours; 
        # <...> empty for now
        # an absolute increase in urine M-protein levels in patients without measurable serum and urine M-protein levels; 
        # <...> empty for now
        # bone marrow plasma cell percentage of at least 10%; 
        # <...> empty for now
        # development of new bone lesions or soft tissue plasmacytomas; 
        # <...> empty for now
        # or development of hypercalcemia.
        # urine M-component with an absolute increase of at least 200 mg per 24 hours; 
        # <...> empty for now
        # an absolute increase in urine M-protein levels in patients without measurable serum and urine M-protein levels; 
        # <...> empty for now
        # bone marrow plasma cell percentage of at least 10%; 
        # <...> empty for now
        # development of new bone lesions or soft tissue plasmacytomas; 
        # <...> empty for now
        # or development of hypercalcemia.
    #print("True PFS\n", true_pfs)
    #print("Average PFS among actual PFS", np.mean(true_pfs[true_pfs>0]))
    #print("Std PFS among actual PFS", np.std(true_pfs[true_pfs>0]))
    #print("Median PFS among actual PFS", np.median(true_pfs[true_pfs>0]))
    #print("Max PFS among actual PFS", np.max(true_pfs[true_pfs>0]))
    #print("Min PFS among actual PFS", np.min(true_pfs[true_pfs>0]))
    return true_pfs
