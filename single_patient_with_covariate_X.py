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
sigma = 0.1
# True parameter values

X = 1

rho_s_population = -0.005
rho_r_population = 0.001
pi_r_population = 0.4
psi_population = 50

theta_expected_rho_s_population = np.log(-rho_s_population)
theta_expected_rho_r_population = np.log(rho_r_population)
theta_expected_pi_r_population  = np.log(pi_r_population/(1-pi_r_population))
print("theta_expected_rho_s_population:", theta_expected_rho_s_population)
print("theta_expected_rho_r_population:", theta_expected_rho_r_population)
print("theta_expected_pi_r_population:",  theta_expected_pi_r_population)

true_beta = np.array([theta_expected_rho_s_population, theta_expected_rho_r_population, theta_expected_pi_r_population])

days_between_measurements = 30 # every X days
number_of_measurements = 50 # for N*X days
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements, number_of_measurements+1)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])
these_parameters = Parameters(Y_0=psi_population, pi_r=pi_r_population, g_r=rho_r_population, g_s=rho_s_population, k_1=0, sigma=sigma)
this_patient = Patient(these_parameters, measurement_times, treatment_history, name="Single_patient")
plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Single_patient", savename="./plots/Bayes_simulated_data/"+"0-Single_patient")

Y = this_patient.Mprotein_values
t = this_patient.measurement_times
#print("Y:", Y)
#print("t:", t)
##############################
single_patient_model = pm.Model()

with single_patient_model:
    sigma = pm.HalfNormal("sigma", sigma=1)

    # beta
    beta = pm.Normal("beta",  mu=true_beta,  sigma=1, shape=3)

    # Latent variables theta
    theta_expected_rho_s = X*beta[0] #pm.Normal("theta_expected_rho_s", mu=X*true_beta[0], sigma=1)
    theta_expected_rho_r = X*beta[1] #pm.Normal("theta_expected_rho_r", mu=X*true_beta[1], sigma=1)
    theta_expected_pi_r  = X*beta[2] #pm.Normal("theta_expected_pi_r",  mu=X*true_beta[2], sigma=1)

    # Transformed latent variables 
    rho_s = -np.exp(theta_expected_rho_s)
    rho_r = np.exp(theta_expected_rho_r)
    pi_r  = 1/(1+np.exp(-theta_expected_pi_r))

    # Psi separate with free uninformative prior 
    psi = pm.Normal("psi", mu=50, sigma=10)

    # Observation model 
    mu_Y = psi * (pi_r* np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma, observed=Y)

with single_patient_model:
    # draw 1000 posterior samples
    idata = pm.sample()

# We can see the first 5 values for the alpha variable in each chain as follows:
#print("Showing data array for sigma:\n", idata.posterior["sigma"].sel(draw=slice(0, 4)))
#print("Showing data array for theta_expected_rho_s:\n", idata.posterior["theta_expected_rho_s"].sel(draw=slice(0, 4)))
#print("Showing data array for theta_expected_rho_r:\n", idata.posterior["theta_expected_rho_r"].sel(draw=slice(0, 4)))
#print("Showing data array for theta_expected_pi_r:\n", idata.posterior["theta_expected_pi_r"].sel(draw=slice(0, 4)))
#print("Showing data array for psi:\n", idata.posterior["psi"].sel(draw=slice(0, 4)))

az.plot_trace(idata, combined=True)
plt.show()
plt.close()
print(az.summary(idata, round_to=2))
