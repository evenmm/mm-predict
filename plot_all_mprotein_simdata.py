from utilities import *
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
SAVEDIR = "./plots/Bayesian_estimates_simdata_comparison/"

# Data generation 
true_sigma_obs = 1
RANDOM_EFFECTS = True
N_patients = 300
P = 5 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter

#true_omega = np.array([0.5, 0.3, 0.5]) # Good without covariate effects
true_omega = np.array([0.5, 0.05, 0.05])
simulate_rho_r_dependancy_on_rho_s = False
#coef_rho_s_rho_r = 0.3 if simulate_rho_r_dependancy_on_rho_s else 0.0
coef_rho_s_rho_r = 0
# Positive correlation between rho_s and rho_r ON THE THETA SCALE. Higher theta_rho_s (faster decline) means higher theta_rho_r (faster relapse)
model_rho_r_dependancy_on_rho_s = simulate_rho_r_dependancy_on_rho_s
print("simulate_rho_r_dependancy_on_rho_s", simulate_rho_r_dependancy_on_rho_s)

max_time = 1 + 15*28 #25*28 in real data
max_time = 1 + 51*28 #25*28 in real data
days_between_measurements = 28
measurement_times = np.array(range(1,max_time+days_between_measurements,days_between_measurements))
M_number_of_measurements = len(measurement_times)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])
DIFFERENT_LENGTHS = False

# 1 Generate simulated patients 
# Put a USUBJID row in X with USUBJID=True
# Removed. Uniform. But backwards compatibility true_omega_for_psi = 0 #0.3 or 0.2
X, patient_dictionary_complete, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_2_0, true_omega, 0, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS, USUBJID=True, simulate_rho_r_dependancy_on_rho_s=simulate_rho_r_dependancy_on_rho_s, coef_rho_s_rho_r=coef_rho_s_rho_r, DIFFERENT_LENGTHS=DIFFERENT_LENGTHS)

true_pfs_complete_patient_dictionary = get_true_pfs_new(patient_dictionary_complete, time_scale=1, M_scale=1)
print("true_pfs_complete_patient_dictionary", true_pfs_complete_patient_dictionary)
# Crop measurements afer progression: 
for ii, patient in enumerate(patient_dictionary_complete.values()):
    if true_pfs_complete_patient_dictionary[ii] > 0:
        patient.Mprotein_values = patient.Mprotein_values[patient.measurement_times <= true_pfs_complete_patient_dictionary[ii]] # + 2*28]
        patient.measurement_times = patient.measurement_times[patient.measurement_times <= true_pfs_complete_patient_dictionary[ii]] # + 2*28]


# Plot all M protein 
"""
N_patients_train = len(train_dict_all)
patient_dictionary_full = deepcopy(train_dict_all)
for ii in range(len(test_dict_all)):
    patient_dictionary_full[ii+N_patients_train] = deepcopy(test_dict_all[ii])
"""
max_measurement_times = max([max(patient.measurement_times) for patient in patient_dictionary_complete.values()])
print("Total number of patients: ", len(patient_dictionary_complete))

fig, ax1 = plt.subplots(1,1,figsize=(10,6))
for patient in patient_dictionary_complete.values():
    ax1.plot(patient.measurement_times, patient.Mprotein_values, "-", lw=1, alpha=0.5)
ax1.set_ylabel("Serum M protein (g/L)")
# xticks
latest_cycle_start = int(np.rint((max_measurement_times-1)//28 + 1))
tick_labels = [1] + [cs for cs in range(6, latest_cycle_start+1, 6)]
tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
ax1.set_xticks(tick_positions, tick_labels)
ax1.set_xlabel("Cycle number")
plt.tight_layout()
plt.savefig(SAVEDIR+"mprotein_all_patients_simdata.png", dpi=300)
plt.show()
plt.close()
