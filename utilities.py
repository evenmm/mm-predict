import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from scipy import optimize
from scipy.optimize import least_squares
from copy import deepcopy

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

drug_dictionary = np.load("drug_dictionary.npy", allow_pickle=True).item()
treatment_to_id_dictionary = np.load("treatment_to_id_dictionary.npy", allow_pickle=True).item()
treatment_id_to_drugs_dictionary = {v: k for k, v in treatment_to_id_dictionary.items()}

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
#drug_dictionary = dict(zip(unique_drugs, drug_ids))
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

class Treatment:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id
    def get_drug_ids(self):
        return [drug_dictionary[drug_name] for drug_name in treatment_id_to_drugs_dictionary[self.id]]

class Patient: 
    def __init__(self, parameters, measurement_times, treatment_history, covariates = []):
        self.measurement_times = measurement_times
        self.treatment_history = treatment_history
        self.observed_values = measure_Mprotein_with_noise(parameters, self.measurement_times, self.treatment_history)
        self.covariates = covariates
    def get_measurement_times(self):
        return self.measurement_times
    def get_treatment_history(self):
        return self.treatment_history
    def get_observed_values(self):
        return self.observed_values

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
    Y_t = params.Y_0
    pi_r_t = params.pi_r
    t_params = Parameters(Y_t, pi_r_t, params.g_r, params.g_s, params.k_1, params.sigma)
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        this_treatment = treatment_history[treat_index]
        if this_treatment.id == 0:
            drug_effect = 0
        elif this_treatment.id == 1:
            drug_effect = t_params.k_1
        else:
            print("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
            sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        
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
        Y_t = recorded_and_endtime_mprotein_values[-1]
        pi_r_t = resistant_mprotein[-1] / (resistant_mprotein[-1] + sensitive_mprotein[-1])
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

#treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
def plot_true_mprotein_with_observations_and_treatments_and_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    observed_values = patient.get_observed_values()
    plotting_times = np.linspace(0, int(measurement_times[-1]), int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    end_of_history = treatment_history[-1].end
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2
    patient_count = 0

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

    ax1.plot(measurement_times, observed_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title("Patient " + str(1))
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("Treatment line")
    ax2.set_yticks(range(maxdrugkey+1))
    ax2.set_yticklabels(range(maxdrugkey+1))
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    if PLOT_ESTIMATES:
        plt.savefig("./patient_truth_and_observations_with_model_fit.pdf")
    else:
        plt.savefig("./patient_truth_and_observations.pdf")
    plt.show()
    plt.close()


#####################################
# Generate data 
#####################################
# Shared parameters
#####################################
global_sigma = 0.1  #Measurement noise
# Drug effects: 
# map id=1 to effect=k_1
# map id=2 to effect=k_2
# ...

## Patient specific parameters
# No drug:   A doubling time of 2 months = 60 days means that X1/X0 = 2   = e^(60*alpha) ==> alpha = log(2)/60   = approx.  0.005 1/days
# With drug: A halving  time of 2 months = 60 days means that X1/X0 = 1/2 = e^(60*alpha) ==> alpha = log(1/2)/60 = approx. -0.005 1/days
# Encoding cost of resistance giving resistant cells a lower base growth rate than sensitive cells 

"""
#####################################
# Patient 1
#####################################
# With drug effect and growth rate parameters:
#parameters_patient_1 = Parameters(Y_0=50, pi_r=0.10, g_r=0.008, g_s=0.010, k_1=0.020, sigma=global_sigma)
# With bulk growth rates on treatment:
#parameters_patient_1 = Parameters(Y_0=50, pi_r=0.01, g_r=0.008, g_s=-0.010, k_1=0.000, sigma=global_sigma)
parameters_patient_1 = Parameters(Y_0=50, pi_r=0.01, g_r=0.040, g_s=-0.200, k_1=0.000, sigma=global_sigma)

# Measure M protein
Mprotein_recording_interval_patient_1 = 10 #every X days
N_Mprotein_measurements_patient_1 = 5 # for N*X days
measurement_times_patient_1 = Mprotein_recording_interval_patient_1 * np.linspace(0,N_Mprotein_measurements_patient_1,N_Mprotein_measurements_patient_1+1)

# Simplest case: Only one treatment period
T_1_p_1 = Treatment(start=0, end=measurement_times_patient_1[-1], id=1)
treatment_history_patient_1 = [T_1_p_1]

patient_1 = Patient(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
patient_1.plot()

print("Measurement times patient 1", measurement_times_patient_1)

noiseless_M_protein_values_patient_1 = measure_Mprotein_naive(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
print("noiseless_M_protein_values_patient_1, Naive method:\n", noiseless_M_protein_values_patient_1)

noiseless_M_protein_values_patient_1 = measure_Mprotein_noiseless(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
print("noiseless_M_protein_values_patient_1, efficient method:\n", noiseless_M_protein_values_patient_1)

observed_M_protein_values_patient_1 = measure_Mprotein_with_noise(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
print("observed_M_protein_values_patient_1, efficient method:\n", observed_M_protein_values_patient_1)

#####################################
# Plot data 
#####################################
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)

#####################################
# Inference
#####################################
# For inferring both growth rates, k_D and sigma:
##      Y_0, pi_r,   g_r,   g_s,   k_1=0.020
#lb = [   0,    0, -0.10, -0.10,  0.00, 1e-6]
#ub = [1000,    1,  0.10,  0.10,  0.50,  5e4]
params_to_be_inferred = np.array([parameters_patient_1.Y_0, parameters_patient_1.pi_r, parameters_patient_1.g_r, parameters_patient_1.g_s])
# For inferring bulk growth rates
#               Y_0, pi_r,   g_r,   g_s,
lb = np.array([   0,    0, -1.00, -2.00])
ub = np.array([1000,    1,  1.00,  2.00])
bounds_Y_0 = (lb[0], ub[0])
bounds_pi_r = (lb[1], ub[1])
bounds_g_r = (lb[2], ub[2])
bounds_g_s = (lb[3], ub[3])
#bounds_Y_0 =     (0, 1000)
#bounds_pi_r =    (0,    1)
#bounds_g_r = (-0.10, 0.10)
#bounds_g_s = (-0.10, 0.10)

all_bounds = (bounds_Y_0, bounds_pi_r, bounds_g_r, bounds_g_s)

def least_squares_objective_function(array_x, measurement_times, treatment_history, observations):
    Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=array_x[1], g_r=array_x[2], g_s=array_x[3], k_1=0.000, sigma=global_sigma)
    predictions = measure_Mprotein_noiseless(Parameter_object_x, measurement_times, treatment_history)
    sumofsquares = np.sum((observations - predictions)**2)
    return sumofsquares

random_samples = np.random.uniform(0,1,len(ub))
x0_0 = lb + np.multiply(random_samples, (ub-lb))

lowest_f_value = np.inf
best_x = np.array([0,0,0,0])
for iteration in range(1000):
    random_samples = np.random.uniform(0,1,len(ub))
    x0 = lb + np.multiply(random_samples, (ub-lb))
    optimization_result = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1), bounds=all_bounds, options={'disp':False})
    if optimization_result.fun < lowest_f_value:
        lowest_f_value = optimization_result.fun
        best_x = optimization_result.x

print("Compare truth with estimate:")
print("True x:", params_to_be_inferred)
print("Inferred x:", best_x)

f_value_at_truth = least_squares_objective_function(params_to_be_inferred, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)
x0_value = least_squares_objective_function(x0_0, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)

print("f value at first x0:", x0_value)
print("f value at truth:", f_value_at_truth)
print("f value at estimate:", lowest_f_value)

estimated_parameters = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=0.000, sigma=global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1, estimated_parameters=estimated_parameters, PLOT_ESTIMATES=True)
"""

#####################################
# Patient 2
#####################################
# With k drug effect and growth rate parameters:
parameters_patient_2 = Parameters(Y_0=50, pi_r=0.10, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)

# Measure M protein
Mprotein_recording_interval_patient_2 = 10 #every X days
N_Mprotein_measurements_patient_2 = 4 # for N*X days
measurement_times_patient_2 = Mprotein_recording_interval_patient_2 * np.linspace(0,N_Mprotein_measurements_patient_2,N_Mprotein_measurements_patient_2+1)

treatment_history_patient_2 = [
    Treatment(start=0, end=measurement_times_patient_2[2], id=1),
    Treatment(start=measurement_times_patient_2[2], end=measurement_times_patient_2[4], id=0),
    #Treatment(start=measurement_times_patient_2[1], end=measurement_times_patient_2[2], id=0),
    #Treatment(start=measurement_times_patient_2[2], end=measurement_times_patient_2[3], id=1),
    #Treatment(start=measurement_times_patient_2[3], end=measurement_times_patient_2[4], id=1),
    #Treatment(start=measurement_times_patient_2[4], end=measurement_times_patient_2[5], id=0),
    #Treatment(start=measurement_times_patient_2[5], end=measurement_times_patient_2[7], id=0),
    #Treatment(start=measurement_times_patient_2[7], end=measurement_times_patient_2[-1], id=1),
    ]

patient_2 = Patient(parameters_patient_2, measurement_times_patient_2, treatment_history_patient_2, covariates = [0])
"""
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_2, patient_2, estimated_parameters=[], PLOT_ESTIMATES=False)
## Inference
# For inferring both k_1 and growth rates
#               Y_0, pi_r,   g_r,   g_s,  k_1
lb = np.array([   0,    0,  0.00,  0.00, 0.00])
ub = np.array([1000,    1,  2.00,  2.00, 2.00])

param_array_patient_2 = parameters_patient_2.to_array_without_sigma()

bounds_Y_0 = (lb[0], ub[0])
bounds_pi_r = (lb[1], ub[1])
bounds_g_r = (lb[2], ub[2])
bounds_g_s = (lb[3], ub[3])
bounds_k_1 = (lb[4], ub[4])
all_bounds = (bounds_Y_0, bounds_pi_r, bounds_g_r, bounds_g_s, bounds_k_1)

def least_squares_objective_function(array_x, patient):
    measurement_times = patient.measurement_times
    treatment_history = patient.treatment_history
    observations = patient.observed_values

    Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=array_x[1], g_r=array_x[2], g_s=array_x[3], k_1=array_x[4], sigma=global_sigma)
    predictions = measure_Mprotein_noiseless(Parameter_object_x, measurement_times, treatment_history)
    sumofsquares = np.sum((observations - predictions)**2)
    return sumofsquares

random_samples = np.random.uniform(0,1,len(ub))
x0_0 = lb + np.multiply(random_samples, (ub-lb))

lowest_f_value = np.inf
best_x = np.array([0,0,0,0,0])
for iteration in range(100000):
    random_samples = np.random.uniform(0,1,len(ub))
    x0 = lb + np.multiply(random_samples, (ub-lb))
    optimization_result = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(patient_2), bounds=all_bounds, options={'disp':False})
    if optimization_result.fun < lowest_f_value:
        lowest_f_value = optimization_result.fun
        best_x = optimization_result.x

print("Compare truth with estimate:")
print("True x:", param_array_patient_2)
print("Inferred x:", best_x)

f_value_at_truth = least_squares_objective_function(param_array_patient_2, patient_2)
x0_value = least_squares_objective_function(x0_0, patient_2)

print("f value at first x0:", x0_value)
print("f value at truth:", f_value_at_truth)
print("f value at estimate:", lowest_f_value)

estimated_parameters = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_2, patient_2, estimated_parameters=estimated_parameters, PLOT_ESTIMATES=True)
"""

#####################################################
# Learn effect of history on drug response parameters 
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
# create dataset

parameters_patient_3 = Parameters(Y_0=50, pi_r=0.10, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)
parameters_patient_4 = Parameters(Y_0=50, pi_r=0.90, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)
parameters_patient_5 = Parameters(Y_0=50, pi_r=0.90, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)
patient_3 = deepcopy(patient_2)

patient_4 = deepcopy(patient_2)
patient_4.covariates = [1]
patient_5 = deepcopy(patient_4)
print(patient_5.covariates)

# n = 4, p = 1, len(Y) = 4
X = np.zeros((4,1))
X[0,:] = patient_2.covariates
X[1,:] = patient_3.covariates
X[2,:] = patient_4.covariates
X[3,:] = patient_5.covariates
print(X)
y = np.zeros((4,4))
y[0,:] = parameters_patient_2.to_array_without_sigma()[1:5]
y[1,:] = parameters_patient_3.to_array_without_sigma()[1:5]
y[2,:] = parameters_patient_4.to_array_without_sigma()[1:5]
y[3,:] = parameters_patient_5.to_array_without_sigma()[1:5]
print(y)
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
test_covariates = [0]
#row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([test_covariates])
# summarize prediction
print(yhat[0])

