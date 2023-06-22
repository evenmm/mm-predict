from utilities import *
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")

##############################
# Load generated data
# true rho_s
picklefile = open('./binaries_and_pickles/Bayesian_true_rho_s_simulation_study', 'rb')
true_rho_s = pickle.load(picklefile)
picklefile.close()
# true rho_r
picklefile = open('./binaries_and_pickles/Bayesian_true_rho_r_simulation_study', 'rb')
true_rho_r = pickle.load(picklefile)
picklefile.close()
# true pi_r
picklefile = open('./binaries_and_pickles/Bayesian_true_pi_r_simulation_study', 'rb')
true_pi_r = pickle.load(picklefile)
picklefile.close()
# true psi_r
picklefile = open('./binaries_and_pickles/Bayesian_true_psi_simulation_study', 'rb')
true_psi = pickle.load(picklefile)
picklefile.close()

# These were specified in data generation, but the X variables will skew these upwards!!!! 
rho_s_population = -0.005
rho_r_population = 0.001
pi_r_population = 0.4
psi_population = 50
# Print population averages
print("Average true_rho_s:", np.mean(true_rho_s))
print("Average true_rho_r:", np.mean(true_rho_r))
print("Average true_pi_r:", np.mean(true_pi_r))
print("Average true_psi:", np.mean(true_psi))

# Load period and patient definitions
picklefile = open('./binaries_and_pickles/Bayesian_patient_dictionary_simulation_study', 'rb')
patient_dictionary = pickle.load(picklefile)
picklefile.close()

# Load df_X_covariates
picklefile = open('./binaries_and_pickles/Bayesian_df_X_covariates_simulation_study', 'rb')
df_X_covariates = pickle.load(picklefile)
picklefile.close()

N = 300 # num individuals 
P = 1 # One effect from HRD on one parameter, and one offset 
K = 3 # num parameters in theta not including psi.

y_pre_padding = [patient.Mprotein_values for _, patient in patient_dictionary.items()]
times_pre_padding = [patient.measurement_times for _, patient in patient_dictionary.items()]
len_y_each_patient = [len(elem) for elem in times_pre_padding]
max_len_y = max(len_y_each_patient)

y = [[np.nan for tt in range(max_len_y)] for ii in range(N)]
times = [[np.nan for tt in range(max_len_y)] for ii in range(N)]
#y = np.full((N, max_len_y), np.inf)
#times = np.full((N, max_len_y), np.inf)
for i in range(N):
    for t in range(len_y_each_patient[i]):
        y[i][t] = y_pre_padding[i][t]
        times[i][t] = times_pre_padding[i][t]
M = len(y[0]) # num observations in y

# Data objects:
# matrix<lower=0>[N, M] y      # observations y
# matrix[N, M] t               # observation times t
# matrix<lower=0>[N, P] x      # covariates x
y = y 
t = times
x = [[elem[1]] for elem in df_X_covariates] # num covariates in x

# Assign the data to a dictionary
MM_data = {
    "N": N,
    "M": M,
    "P": P,
    "K": K,
    "y": y,
    "t": t,
    "x": x,
}





##############################
hier_bayes_model = pm.Model()

with hier_bayes_model:

    # Hyperpriors
    omega = pm.Normal("omega", mu=0, sigma=10, shape=K)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10, shape=3)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=3)
    # Future: 
    #beta_rho_s = pm.Normal("beta_rho_s", mu=0, sigm
    a=10, shape=P)
    #beta_rho_r = pm.Normal("beta_rho_r", mu=0, sigma=10, shape=P)
    #beta_pi_r = pm.Normal("beta_pi_r", mu=0, sigma=10, shape=P)
    ##beta = pm.Normal("beta", mu=0, sigma=10, shape=(K,P))

    # Expected values for theta 
    theta_expected_1 = alpha[0] + beta[0] * X1
    theta_expected_2 = alpha[1] + beta[1] * X1
    theta_expected_3 = alpha[2] + beta[2] * X1
    # Future: 
    #theta_1 = pm.Normal("theta_1", mu=theta_expected_1, sigma=omega_1)
    #theta_2 = pm.Normal("theta_2", mu=theta_expected_2, sigma=omega_2)
    #theta_3 = pm.Normal("theta_3", mu=theta_expected_3, sigma=omega_3)
    #theta = alpha + beta[0] * X

    # Patient specific parameters
    rho_s = -exp(theta_expected_1)
    rho_r = exp(theta_expected_2)
    pi_r  = 1/(1+exp(-theta_expected_3))
    psi = pm.Normal("psi", mu=50, sigma=10, shape=N)

    # Likelihood (sampling distribution) of observations
    mu_Y = psi[i] * (pi_r[i]* exp(rho_r[i]*t[i][j]) + (1-pi_r[i])*exp(rho_s[i]*t[i][j]))

    Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma, observed=Y)

with hier_bayes_model:
    # draw 1000 posterior samples
    idata = pm.sample()

# We can see the first 5 values for the alpha variable in each chain as follows:
print("Showing data array for alpha:\n", idata.posterior["alpha"].sel(draw=slice(0, 4)))

az.plot_trace(idata, combined=True)
plt.show()
plt.close()
print(az.summary(idata, round_to=2))
