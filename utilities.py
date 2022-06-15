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

def isNaN(string):
    return string != string
def Sort(sub_li):
    return(sorted(sub_li, key = lambda x: x[1]))

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
drug_dictionary_OSLO = np.load("./binaries_and_pickles/drug_dictionary_OSLO.npy", allow_pickle=True).item()
drug_dictionary_COMMPASS = np.load("./binaries_and_pickles/drug_dictionary_COMMPASS.npy", allow_pickle=True).item()
for key, value in drug_dictionary_COMMPASS.items():
    if key not in drug_dictionary_OSLO.keys():
        drug_dictionary_OSLO[key] = (max(drug_dictionary_OSLO.values())+1)
# Join the two drug dictionaries to get a complete drug dictionary 
drug_dictionary = drug_dictionary_OSLO
np.save("./binaries_and_pickles/drug_dictionary.npy", drug_dictionary)
drug_id_to_name_dictionary = {v: k for k, v in drug_dictionary.items()}
treatment_to_id_dictionary = np.load("./binaries_and_pickles/treatment_to_id_dictionary_OSLO.npy", allow_pickle=True).item()
#treatment_to_id_dictionary = np.load("./binaries_and_pickles/treatment_to_id_dictionary_COMMPASS.npy", allow_pickle=True).item()
treatment_id_to_drugs_dictionary = {v: k for k, v in treatment_to_id_dictionary.items()}

def get_drug_names_from_treatment_id_COMMPASS(treatment_id, treatment_id_to_drugs_dictionary_COMMPASS):
    drug_set = treatment_id_to_drugs_dictionary_COMMPASS[treatment_id]
    drug_names = [drug_id_to_name_dictionary[elem] for elem in drug_set]
    return drug_names

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
        self.Y_0 = Y_0
        self.pi_r = pi_r
        self.g_r = g_r
        self.g_s = g_s
        self.k_1 = k_1
        self.sigma = sigma
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
    def __init__(self, parameters, measurement_times, treatment_history, covariates = []):
        self.measurement_times = measurement_times
        self.treatment_history = treatment_history
        self.Mprotein_values = measure_Mprotein_with_noise(parameters, self.measurement_times, self.treatment_history)
        self.covariates = covariates
    def get_measurement_times(self):
        return self.measurement_times
    def get_treatment_history(self):
        return self.treatment_history
    def get_Mprotein_values(self):
        return self.Mprotein_values

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

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_noiseless(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times)
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

#####################################
# Plotting
#####################################
#treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
def plot_true_mprotein_with_observations_and_treatments_and_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    first_time = min(measurement_times[0], treatment_history[0].start)
    plotting_times = np.linspace(first_time, int(measurement_times[-1]), int((measurement_times[-1]+1)*10))
    
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
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("Treatment line. max="+str(maxdrugkey))
    ax2.set_yticks(range(maxdrugkey+1))
    ax2.set_yticklabels(range(maxdrugkey+1))
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    if savename != 0:
        plt.savefig(savename)
    else:
        if PLOT_ESTIMATES:
            plt.savefig("./patient_truth_and_observations_with_model_fit"+plot_title+".pdf")
        else:
            plt.savefig("./patient_truth_and_observations"+plot_title+".pdf")
    #plt.show()
    plt.close()

def plot_treatment_region_with_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = max(treatment_history[-1].end, int(measurement_times[-1]))
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
    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
    # Plot total M protein
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
    ax1.set_ylabel("Serum Mprotein (g/dL)")
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
    plt.savefig(savename)
    #plt.show()
    plt.close()


def plot_to_compare_estimated_and_predicted_drug_dynamics(true_parameters, predicted_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = max(treatment_history[-1].end, int(measurement_times[-1]))
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
    ax1.set_ylabel("Serum Mprotein (g/dL)")
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
    plt.savefig(savename)
    #plt.show()
    plt.close()

#####################################
# Inference
#####################################
global_sigma = 0.1  #Measurement noise

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

def estimate_drug_response_parameters(patient, lb, ub, N_iterations=10000):
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
    with Pool(15) as pool:
        optim_results = pool.map(get_optimization_result,args)
    
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
