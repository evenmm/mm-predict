import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from scipy import optimize
from scipy.optimize import least_squares

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

#####################################
# Classes, functions
#####################################
class Parameters: 
    def __init__(self, Y_0, pi_r, g_r, g_s, k_1, sigma):
        self.Y_0 = Y_0
        self.pi_r = pi_r
        self.g_r = g_r
        self.g_s = g_s
        self.k_1 = k_1
        self.sigma = sigma

class Treatment:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id

# Efficient implementation 
# Simulates M protein value at times [t + delta_T]_i
# Y_t is the M protein level at start of time interval
def generative_model(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values) + Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_noiseless(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times)
    Y_t = params.Y_0
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        this_treatment = treatment_history[treat_index]
        if this_treatment.id == 0:
            drug_effect = 0
        elif this_treatment.id == 1:
            drug_effect = params.k_1
        else:
            print("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
            sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        
        # Filter that selects measurement times occuring while on this treatment line
        correct_times = (measurement_times >= this_treatment.start) & (measurement_times <= this_treatment.end)
        
        delta_T_values = measurement_times[correct_times] - this_treatment.start
        # Add delta T for (end - start) to keep track of Mprotein at end of treatment
        delta_T_values = np.concatenate((delta_T_values, np.array([this_treatment.end - this_treatment.start])))

        # Calculate Mprotein values
        recorded_and_endtime_mprotein_values = generative_model(Y_t, params, delta_T_values, drug_effect)
        # Assign M protein values for measurement times that are in this treatment period
        Mprotein_values[correct_times] = recorded_and_endtime_mprotein_values[0:-1]
        # Store Mprotein value at the end of this treatment:
        Y_t = recorded_and_endtime_mprotein_values[-1]
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
def plot_true_mprotein_with_observations_and_treatments_and_estimate(true_parameters, measurement_times, treatment_history, M_protein_observations, estimated_parameters=[], PLOT_ESTIMATES=False):
    # Resolution of 10 points per day, plotting 10 days beyond last treatment
    #plotting_times = np.linspace(0, int(measurement_times[-1]+10), int((measurement_times[-1]+10+1)*10))
    plotting_times = np.linspace(0, int(measurement_times[-1]), int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    end_of_history = treatment_history[-1].end
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    if PLOT_ESTIMATES:
        estimated_mprotein_values = measure_Mprotein_noiseless(estimated_parameters, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2
    patient_count = 0

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k')
    if PLOT_ESTIMATES:
        # Plot estimtated Mprotein line
        ax1.plot(plotting_times, estimated_mprotein_values, linestyle='-', marker='', zorder=3, color='r')

    ax1.plot(measurement_times_patient_1, M_protein_observations, linestyle='', marker='x', zorder=3, color='k')

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
    fig.tight_layout()
    if PLOT_ESTIMATES:
        plt.savefig("./patient_truth_and_observations_with_model_fit.pdf")
    else:
        plt.savefig("./patient_truth_and_observations.pdf")
    plt.show()
    plt.close()


#####################################
# Parameters
#####################################
## Shared parameters
global_sigma = 1  #Measurement noise
# Drug effects: 
# map id=1 to effect=k_1
# map id=2 to effect=k_2
# ...

## Patient specific parameters
# No drug:   A doubling time of 2 months = 60 days means that X1/X0 = 2   = e^(60*alpha) ==> alpha = log(2)/60   = approx.  0.005 1/days
# With drug: A halving  time of 2 months = 60 days means that X1/X0 = 1/2 = e^(60*alpha) ==> alpha = log(1/2)/60 = approx. -0.005 1/days
# Encoding cost of resistance giving resistant cells a lower base growth rate than sensitive cells 

# With drug effect and growth rate parameters:
#parameters_patient_1 = Parameters(Y_0=50, pi_r=0.10, g_r=0.008, g_s=0.010, k_1=0.020, sigma=global_sigma)
# With bulk growth rates on treatment:
parameters_patient_1 = Parameters(Y_0=50, pi_r=0.20, g_r=0.040, g_s=-0.200, k_1=0.000, sigma=global_sigma)
#parameters_patient_1 = Parameters(Y_0=50, pi_r=0.01, g_r=0.008, g_s=-0.010, k_1=0.000, sigma=global_sigma)

# Measure M protein
Mprotein_recording_interval = 10 #every X days
N_Mprotein_measurements = 5 # for N*X days
measurement_times_patient_1 = Mprotein_recording_interval * np.linspace(0,N_Mprotein_measurements,N_Mprotein_measurements+1)

#####################################
# Generate data 
#####################################
# Simplest case: Only one treatment period
T_1_p_1 = Treatment(start=0, end=measurement_times_patient_1[-1], id=1)
treatment_history_patient_1 = [T_1_p_1]

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
lb = np.array([   0,    0, -1.00, -1.00])
ub = np.array([1000,    1,  1.00,  1.00])
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

