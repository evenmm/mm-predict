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
x1_mean = 0
x1_std = 0.5
true_x1 = np.random.normal(x1_mean, x1_std, size=N_patients)
# These are not the means, but the parameters when x1 = 0 (medians):
rho_s_population = -0.005
rho_r_population = 0.001
pi_r_population = 0.4
psi_population = 50
theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

true_omega = np.array([0.05, 0.10, 0.15])
true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
true_beta = np.array([0.9,1.0,1.1])

print("true_alpha[0]:", true_alpha[0])
print("true_alpha[1]:", true_alpha[1])
print("true_alpha[2]:", true_alpha[2])
print("true_beta[0]: ", true_beta[0])
print("true_beta[1]: ", true_beta[1])
print("true_beta[2]: ", true_beta[2])

days_between_measurements = 30
number_of_measurements = 50
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements, number_of_measurements+1)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

true_psi = np.exp(np.random.normal(np.log(psi_population),0.1,size=N_patients))
print("true_psi[-5:-1]:", true_psi[-5:-1])
patient_dictionary = {}
true_theta_rho_s = np.zeros(N_patients)
true_theta_rho_r = np.zeros(N_patients)
true_theta_pi_r = np.zeros(N_patients)
true_rho_s = np.zeros(N_patients)
true_rho_r = np.zeros(N_patients)
true_pi_r = np.zeros(N_patients)
# Generate data
for training_instance_id in range(N_patients):
    x1 = true_x1[training_instance_id]
    expected_theta_1_patient_i = true_alpha[0] + true_beta[0]*x1
    expected_theta_2_patient_i = true_alpha[1] + true_beta[1]*x1
    expected_theta_3_patient_i = true_alpha[2] + true_beta[2]*x1

    # Deterministic
    #theta_1_patient_i = expected_theta_1_patient_i
    #theta_2_patient_i = expected_theta_2_patient_i
    #theta_3_patient_i = expected_theta_3_patient_i
    # Patient specific noise / deviation from X effects
    theta_1_patient_i = np.random.normal(expected_theta_1_patient_i, true_omega[0])
    theta_2_patient_i = np.random.normal(expected_theta_2_patient_i, true_omega[1])
    theta_3_patient_i = np.random.normal(expected_theta_3_patient_i, true_omega[2])
    true_theta_rho_s[training_instance_id] = theta_1_patient_i
    true_theta_rho_r[training_instance_id] = theta_2_patient_i
    true_theta_pi_r[training_instance_id] = theta_3_patient_i

    # Transform thetas into parameters
    rho_s_patient_i = - np.exp(theta_1_patient_i)
    rho_r_patient_i = np.exp(theta_2_patient_i)
    pi_r_patient_i = 1/(1+np.exp(-theta_3_patient_i)) # sigmoid 
    psi_patient_i = true_psi[training_instance_id]
    #print("x1   ;", x1)
    #print("rho_s:", rho_s_patient_i)
    #print("rho_r:", rho_r_patient_i)
    #print("pi_r :", pi_r_patient_i)
    #print("psi  :", psi_patient_i)
    true_rho_s[training_instance_id] = rho_s_patient_i
    true_rho_r[training_instance_id] = rho_r_patient_i
    true_pi_r[training_instance_id] = pi_r_patient_i
    true_psi[training_instance_id] = psi_patient_i

    these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=true_sigma)
    this_patient = Patient(these_parameters, measurement_times, treatment_history, name=str(training_instance_id))
    patient_dictionary[training_instance_id] = this_patient
    #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id))
print("true_theta_rho_s:\n", true_theta_rho_s)
print("true_theta_rho_r:\n", true_theta_rho_r)
print("true_theta_pi_r:\n", true_theta_pi_r)
print("\ntrue_rho_s:\n", true_rho_s)
print("true_rho_r:\n", true_rho_r)
print("true_pi_r:\n", true_pi_r)
print("true_psi:\n", true_psi)

#X = [[elem[1]] for elem in df_X_covariates]
X = np.array([[elem] for elem in true_x1])
Y = np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()])
t = np.array([patient.measurement_times for _, patient in patient_dictionary.items()])
yi0 = np.array([[patient.Mprotein_values[0]] for _, patient in patient_dictionary.items()])
#print("Y:", Y)
#print("t:", t)
print("Done generating data")
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("t.shape:", t.shape)
print("Desired psi.shape: (N_patients,1) = ", X.shape)
##############################
multiple_patients_model = pm.Model()

with multiple_patients_model:
    # Observation noise (std)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # alpha
    alpha = pm.Normal("alpha",  mu=true_alpha,  sigma=1, shape=3)
    # beta
    beta = pm.Normal("beta",  mu=true_beta,  sigma=1, shape=3)

    # Latent variables theta
    #theta_rho_s = alpha[0] + X*beta[0] # Deterministically determined by x, beta, alpha
    #theta_rho_r = alpha[1] + X*beta[1] # Deterministically determined by x, beta, alpha
    #theta_pi_r  = alpha[2] + X*beta[2] # Deterministically determined by x, beta, alpha
    omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
    theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + X*beta[0], sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
    theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + X*beta[1], sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
    theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + X*beta[2], sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

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
plt.show()
az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta.png")
plt.show()
az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_s.png")
plt.show()
az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_r.png")
plt.show()
az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_pi_r.png")
plt.show()
