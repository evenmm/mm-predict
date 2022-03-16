import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

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
def plot_true_mprotein_with_observations_and_treatments(true_parameters, measurement_times, treatment_history, M_protein_observations):
    # Resolution of 10 points per day, plotting 10 days beyond last treatment
    #plotting_times = np.linspace(0, int(measurement_times[-1]+10), int((measurement_times[-1]+10+1)*10))
    plotting_times = np.linspace(0, int(measurement_times[-1]), int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    end_of_history = treatment_history[-1].end
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    print(plotting_times)
    print(plotting_mprotein_values)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2
    patient_count = 0

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k')

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
    plt.savefig("./patient_truth_and_observations.pdf")
    plt.show()
    plt.close()

#def least_squares_objective_function(observations, predictions):
    


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
parameters_patient_1 = Parameters(Y_0=50, pi_r=0.10, g_r=0.008, g_s=0.010, k_1=0.020, sigma=global_sigma)

# Measure M protein
Mprotein_recording_interval = 10 #every X days
N_Mprotein_measurements = 36 # for N*X days
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
plot_true_mprotein_with_observations_and_treatments(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)

#####################################
# Inference
#####################################
#MLE_estimates = 

