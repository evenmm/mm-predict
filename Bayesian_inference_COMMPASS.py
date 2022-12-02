from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import aesara.tensor as at
from create_training_instance_dictionary_with_covariates import *
from feature_extraction import *
from sample_from_full_model import *
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
##############################
# Load data
# Load dummy_patient_dict
#patient_dictionary = np.load("./binaries_and_pickles/dummy_patient_dict.npy", allow_pickle=True).item()
M_number_of_measurements = 4
patient_dictionary, training_instance_dict = create_training_instance_dictionary_with_covariates(minimum_number_of_measurements=M_number_of_measurements, global_treatment_id_list = [1,2,3,7,10,13,15,16], verbose=False)
# Dimensions: 
# y: M_max * N
# t: M_max * N
# X: P * N
# Subset data
# Create a training_instance dictionary with covariates and M proteins only in the period of interest. 
#   Idea 1: The drug during the treatment is the only X. Shows you the drug's effect on the mean growth rates. 
#   Idea 2: For each drug, find which factors determine the response. 

#for name, patient in patient_dictionary.items():
#    plot_true_mprotein_with_observations_and_treatments_and_estimate(Parameters(0.1, 0.1, 0.001, -0.001, 0, 0.1), patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(name), savename="./plots/Bayes_simulated_data/COMMPASS/"+str(name))

# This could be a function dict, Y, t = standardize_input_from_dictionary: 
N_patients = len(patient_dictionary)
y_pre_padding = np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()])
#y_pre_padding = max(y_pre_padding,0)
times_pre_padding = np.array([patient.measurement_times for _, patient in patient_dictionary.items()])
times_pre_padding = [t_list-t_list[0] for t_list in times_pre_padding]# Account for nonzero time 0
len_y_each_patient = np.array([len(elem) for elem in times_pre_padding])
max_len_y = max(len_y_each_patient)
y = np.array([[np.nan for tt in range(max_len_y)] for ii in range(N_patients)])
times = np.array([[np.nan for tt in range(max_len_y)] for ii in range(N_patients)])
for i in range(N_patients):
    for t in range(len_y_each_patient[i]):
        y[i][t] = y_pre_padding[i][t]
        times[i][t] = times_pre_padding[i][t]

# Use only fully observed part of data to get fully observed y and t: 
# Scale up Y to get it on a scale further away from zero
y = 100*np.array([elem[0:M_number_of_measurements] for elem in y])
times = np.array([elem[0:M_number_of_measurements] for elem in times])

# y and times are cropped: Update the patient dictionary 
dummy_patient_dict  = {}
for training_instance_id in range(0, N_patients):
    dummy_patient_dict[training_instance_id] = patient_dictionary[training_instance_id]
    dummy_patient_dict[training_instance_id].measurement_times = times[training_instance_id]
    dummy_patient_dict[training_instance_id].Mprotein_values = y[training_instance_id]
patient_dictionary = dummy_patient_dict

# Keep only patients that are in EHR data: 
COMMPASS_current_name_list = [elem[0] for elem in training_instance_dict.values()]
df_EHR = pd.read_excel('./COMMPASS_data/220615_commpass_clinical_genomic_annotated_EHR.xlsx')
EHR_name_list = [elem.replace("_1_BM" ,"", 1) for elem in df_EHR.loc[:,"sample"]]
NEW_TRAIN_ID = 0
new_patient_dictionary = {}
new_training_instance_dict = {}
for training_instance_id, patient in patient_dictionary.items(): # Dummy dictionary has training_instance_id as key
    this_name = COMMPASS_current_name_list[training_instance_id]
    if this_name in EHR_name_list: 
        new_patient_dictionary[NEW_TRAIN_ID] = patient_dictionary[training_instance_id] # equal to: "= patient"
        new_training_instance_dict[NEW_TRAIN_ID] = training_instance_dict[training_instance_id]
        NEW_TRAIN_ID = NEW_TRAIN_ID + 1
N_patients = NEW_TRAIN_ID + 1
# This resets from "patient_dictionary, training_instance_dict = create_training_instance_dictionary_with_covariates"
patient_dictionary = new_patient_dictionary
training_instance_dict = new_training_instance_dict

X = feature_extraction(training_instance_dict)
_, P = X.shape
psi_prior="normal"
N_samples = 3000
N_tuning = 3000
target_accept = 0.99
max_treedepth = 10
FUNNEL_REPARAMETRIZATION = True
name = "COMMPASS_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_patients_"+str(N_patients)+"_psi_prior_"+psi_prior+"_N_samples_"+str(N_samples)+"_N_tuning_"+str(N_tuning)+"_target_accept_"+str(target_accept)+"_max_treedepth_"+str(max_treedepth)+"_FUNNEL_REPARAMETRIZATION_"+str(FUNNEL_REPARAMETRIZATION)
print("Running"+name)
idata = sample_from_full_model(X, patient_dictionary, name, N_samples=N_samples, N_tuning=N_tuning, target_accept=target_accept, max_treedepth=max_treedepth, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION)

print("Done sampling")
az.plot_trace(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma_obs'), combined=True)
plt.savefig("./plots/posterior_plots/"+name+"-plot_posterior_group_parameters.png")
plt.close()

if psi_prior=="lognormal":
    az.plot_trace(idata, var_names=('xi'), combined=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_posterior_group_parameters_xi.png")
    plt.close()

az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
plt.savefig("./plots/posterior_plots/"+name+"-plot_posterior_individual_parameters.png")
plt.close()
# Test of exploration 
az.plot_energy(idata)
plt.savefig("./plots/posterior_plots/"+name+"-plot_energy.png")
plt.close()
# Plot of coefficients
az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_alpha.png")
az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True, rope=(0,0))
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_beta_rho_s.png")
plt.close()
az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True, rope=(0,0))
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_beta_rho_r.png")
plt.close()
az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True, rope=(0,0))
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_beta_pi_r.png")
plt.close()
az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_theta_rho_s.png")
plt.close()
az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_theta_rho_r.png")
plt.close()
az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_theta_pi_r.png")
plt.close()
