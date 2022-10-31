from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
##############################
# Generate data
true_sigma = 0.1
N_patients = 100

# True parameter values
P = 2 # Number of covariates
X_mean = np.repeat(0,P)
X_std = np.repeat(0.5,P)
X = np.random.normal(X_mean, X_std, size=(N_patients,P))
# These are the true parameters for x1 = 0 (median):
rho_s_population = -0.005
rho_r_population = 0.001
pi_r_population = 0.4
psi_population = 50
theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

true_omega = np.array([0.05, 0.10, 0.15])
true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
true_beta_rho_s = np.array([0.8, 0.9])
true_beta_rho_r = np.array([0.7, 1.0])
true_beta_pi_r = np.array([0.5, 1.1])

print("true_alpha[0]:", true_alpha[0])
print("true_alpha[1]:", true_alpha[1])
print("true_alpha[2]:", true_alpha[2])
print("true_beta_rho_s: ", true_beta_rho_s)
print("true_beta_rho_r: ", true_beta_rho_r)
print("true_beta_pi_r: ", true_beta_pi_r)

days_between_measurements = 300
number_of_measurements = 5
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements, number_of_measurements+1)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s), (N_patients,1))
expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients,1))
expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients,1))

# Deterministic
#true_theta_rho_s = expected_theta_1
#true_theta_rho_r = expected_theta_2
#true_theta_pi_r  = expected_theta_3
# Patient specific noise / deviation from X effects
true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])

true_rho_s = - np.exp(true_theta_rho_s)
true_rho_r = np.exp(true_theta_rho_r)
true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
true_psi = np.exp(np.random.normal(np.log(psi_population),0.1,size=N_patients))
print("true_psi[-5:-1]:", true_psi[-5:-1])
patient_dictionary = {}
for training_instance_id in range(N_patients):
    psi_patient_i   = true_psi[training_instance_id]
    pi_r_patient_i  = true_pi_r[training_instance_id]
    rho_r_patient_i = true_rho_r[training_instance_id]
    rho_s_patient_i = true_rho_s[training_instance_id]
    these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=true_sigma)
    this_patient = Patient(these_parameters, measurement_times, treatment_history, name=str(training_instance_id))
    patient_dictionary[training_instance_id] = this_patient
    #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id))
#print("true_theta_rho_s:\n", true_theta_rho_s)
#print("true_theta_rho_r:\n", true_theta_rho_r)
#print("true_theta_pi_r:\n", true_theta_pi_r)
#print("\ntrue_rho_s:\n", true_rho_s)
#print("true_rho_r:\n", true_rho_r)
#print("true_pi_r:\n", true_pi_r)
#print("true_psi:\n", true_psi)

Y = np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()])
t = np.array([patient.measurement_times for _, patient in patient_dictionary.items()])
yi0 = np.array([[patient.Mprotein_values[0]] for _, patient in patient_dictionary.items()])
print("Done generating data")
##############################
multiple_patients_model = pm.Model()
import aesara.tensor as at
with multiple_patients_model:
    # Observation noise (std)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # alpha
    alpha = pm.Normal("alpha",  mu=true_alpha,  sigma=1, shape=3)
    # beta
    beta_rho_s = pm.Normal("beta_rho_s", mu=0, sigma=1, shape=(P,1))
    beta_rho_r = pm.Normal("beta_rho_r", mu=0, sigma=1, shape=(P,1))
    beta_pi_r = pm.Normal("beta_pi_r", mu=0, sigma=1, shape=(P,1))

    # Latent variables theta
    #theta_rho_s = alpha[0] + pm.math.dot(X, beta_rho_s) # Deterministically determined by x, beta, alpha
    #theta_rho_r = alpha[1] + pm.math.dot(X, beta_rho_r) # Deterministically determined by x, beta, alpha
    #theta_pi_r  = alpha[2] + pm.math.dot(X, beta_pi_r) # Deterministically determined by x, beta, alpha
    omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
    theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + pm.math.dot(X, beta_rho_s), sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
    theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + pm.math.dot(X, beta_rho_r), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
    theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + pm.math.dot(X, beta_pi_r),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

    # Transformed latent variables 
    rho_s = -np.exp(theta_rho_s)
    rho_r = np.exp(theta_rho_r)
    pi_r  = 1/(1+np.exp(-theta_pi_r))

    # psi: True M protein at time 0
    #psi = pm.Normal("psi", mu=psi_population, sigma=10, shape=(N_patients,1)) # Informative on population level
    psi = pm.Normal("psi", mu=yi0, sigma=sigma, shape=(N_patients,1)) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 

    # Observation model 
    mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma, observed=Y)

with multiple_patients_model:
    # draw 1000 posterior samples
    idata = pm.sample()

# We can see the first 5 values for the alpha variable in each chain as follows:
#print("Showing data array for beta:\n", idata.posterior["beta"].sel(draw=slice(0, 4)))
#print("Showing data array for sigma:\n", idata.posterior["sigma"].sel(draw=slice(0, 4)))
#print("Showing data array for theta_rho_s:\n", idata.posterior["theta_rho_s"].sel(draw=slice(0, 4)))
#print("Showing data array for theta_rho_r:\n", idata.posterior["theta_rho_r"].sel(draw=slice(0, 4)))
#print("Showing data array for theta_pi_r:\n", idata.posterior["theta_pi_r"].sel(draw=slice(0, 4)))
#print("Showing data array for psi:\n", idata.posterior["psi"].sel(draw=slice(0, 4)))

az.plot_trace(idata, combined=True)
plt.savefig("./plots/posterior_plots/plot_posterior.png")
plt.show()
plt.close()
print(az.summary(idata, round_to=2))

# Test of exploration 
az.plot_energy(idata)
plt.savefig("./plots/posterior_plots/plot_energy.png")
plt.show()
# Plot of coefficients
az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_alpha.png")
#plt.show()
az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta.png")
#plt.show()
az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta.png")
#plt.show()
az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta.png")
#plt.show()
az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_s.png")
#plt.show()
az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_r.png")
#plt.show()
az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_pi_r.png")
#plt.show()
