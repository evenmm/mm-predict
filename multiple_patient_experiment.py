from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import aesara.tensor as at
from sample_from_full_model import *
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
##############################
# Generate data
true_sigma = 0.1
N_patients = 100

# True parameter values
P = 6 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
X_mean = np.repeat(0,P)
X_std = np.repeat(0.5,P)
X = np.random.normal(X_mean, X_std, size=(N_patients,P))
X = pd.DataFrame(X, columns = ["Covariate "+str(ii+1) for ii in range(P)])
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
true_beta_rho_s = np.zeros(P)
true_beta_rho_s[0] = 0.8
true_beta_rho_s[1] = 0.9
true_beta_rho_r = np.zeros(P)
true_beta_rho_r[0] = 0.7
true_beta_rho_r[1] = 1.0
true_beta_pi_r = np.zeros(P)
true_beta_pi_r[0] = 0.0
true_beta_pi_r[1] = 1.1

print("true_alpha[0]:", true_alpha[0])
print("true_alpha[1]:", true_alpha[1])
print("true_alpha[2]:", true_alpha[2])
print("true_beta_rho_s: ", true_beta_rho_s)
print("true_beta_rho_r: ", true_beta_rho_r)
print("true_beta_pi_r: ", true_beta_pi_r)

number_of_measurements = 5
days_between_measurements = int(1500/number_of_measurements)
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements-1, number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s), (N_patients,1))
expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients,1))
expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients,1))

# Patient specific noise / deviation from X effects
true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])
# To generate the data, we employ a "fourth omega" for psi. But since we do not explain theta_psi by a linear predictor
#    , we instead estimate xi in the MCMC, which is the standard deviation of psi_i^0 from y_i1. 
true_omega_for_psi = 0.1
true_theta_psi = np.random.normal(np.log(psi_population), true_omega_for_psi, size=N_patients)
print("true_theta_rho_s[0:5]:\n", true_theta_rho_s[0:5])
print("true_theta_rho_r[0:5]:\n", true_theta_rho_r[0:5])
print("true_theta_pi_r[0:5]:\n", true_theta_pi_r[0:5])
print("true_theta_psi[0:5]:\n", true_theta_psi[0:5])

true_rho_s = - np.exp(true_theta_rho_s)
true_rho_r = np.exp(true_theta_rho_r)
true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
true_psi = np.exp(true_theta_psi)
print("true_psi[0:5]:", true_psi[0:5])
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

Y = np.transpose(np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()]))
t = np.transpose(np.array([patient.measurement_times for _, patient in patient_dictionary.items()]))
yi0 = np.array([patient.Mprotein_values[0] for _, patient in patient_dictionary.items()])
yi0 = np.maximum(yi0, 1e-5)
#print("Y:\n", Y)
#print("t:\n", t)
#print("yi0:\n", yi0)
#print("X:\n", X)
X_not_transformed = X.copy()
print(X_not_transformed.columns.values)
X = X.T
print("Shapes:")
print("Y:", Y.shape)
print("t:", t.shape)
print("yi0:", yi0.shape)
print("X:", X.shape)
print("Done generating data")
##############################
idata = sample_from_full_model(X_not_transformed, patient_dictionary, name="", psi_prior="lognormal")

# We can see the first 5 values for the alpha variable in each chain as follows:
#print("Showing data array for alpha:\n", idata.posterior["alpha"].sel(draw=slice(0, 4)))

print(az.summary(idata, var_names=['theta_rho_s'], round_to=4)) #, 'rho_s', 'rho_r', 'pi_r'], round_to=4))
#print(az.summary(idata, var_names=['theta_rho_s'], round_to=4, stat_focus="median")) #, 'rho_s', 'rho_r', 'pi_r'], round_to=4), stat_focus="median")
print(az.summary(idata, var_names=['theta_rho_r'], round_to=4)) #, 'rho_s', 'rho_r', 'pi_r'], round_to=4))
#print(az.summary(idata, var_names=['theta_rho_r'], round_to=4, stat_focus="median")) #, 'rho_s', 'rho_r', 'pi_r'], round_to=4), stat_focus="median")
print(az.summary(idata, var_names=['theta_pi_r'], round_to=4)) #, 'rho_s', 'rho_r', 'pi_r'], round_to=4))
#print(az.summary(idata, var_names=['theta_pi_r'], round_to=4, stat_focus="median")) #, 'rho_s', 'rho_r', 'pi_r'], round_to=4), stat_focus="median")
print(az.summary(idata, var_names=['alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma'], round_to=4))
print(az.summary(idata, var_names=['alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma'], round_to=4, stat_focus="median"))

print("Access posterior predictive samples directly:")
print('alpha:', np.ravel(idata.posterior['alpha']))
print('beta_rho_s:', np.ravel(idata.posterior['beta_rho_s']))
print('beta_rho_r:', np.ravel(idata.posterior['beta_rho_r']))
print('beta_pi_r:', np.ravel(idata.posterior['beta_pi_r']))
print('omega:', np.ravel(idata.posterior['omega']))
print('sigma:', np.ravel(idata.posterior['sigma']))

lines = [('alpha', {}, true_alpha), ('beta_rho_s', {}, true_beta_rho_s), ('beta_rho_r', {}, true_beta_rho_r), ('beta_pi_r', {}, true_beta_pi_r), ('omega', {}, true_omega), ('sigma', {}, true_sigma)]
az.plot_trace(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma'), lines=lines, combined=True)
plt.savefig("./plots/posterior_plots/plot_posterior_group_parameters.png")
plt.show()
plt.close()

az.plot_trace(idata, var_names=('xi'), combined=True)
plt.savefig("./plots/posterior_plots/plot_posterior_group_parameters_xi.png")
plt.show()
plt.close()

lines = [('theta_rho_s', {}, true_theta_rho_s), ('theta_rho_r', {}, true_theta_rho_r), ('theta_pi_r', {}, true_theta_pi_r), ('rho_s', {}, true_rho_s), ('rho_r', {}, true_rho_r), ('pi_r', {}, true_pi_r)]
az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), lines=lines, combined=True)
plt.savefig("./plots/posterior_plots/plot_posterior_individual_parameters.png")
#plt.show()
plt.close()

# Test of exploration 
az.plot_energy(idata)
plt.savefig("./plots/posterior_plots/plot_energy.png")
#plt.show()
plt.close()
# Plot of coefficients
fig, (ax,) = az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_alpha, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_alpha.png")
#plt.show()
fig, (ax,) = az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_beta_rho_s, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_beta_rho_s.png")
#plt.show()
plt.close()
fig, (ax,) = az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_beta_rho_r, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_beta_rho_r.png")
#plt.show()
plt.close()
fig, (ax,) = az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_beta_pi_r, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_beta_pi_r.png")
#plt.show()
plt.close()
fig, (ax,) = az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_theta_rho_s, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_s.png")
#plt.show()
plt.close()
fig, (ax,) = az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_theta_rho_r, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_r.png")
#plt.show()
plt.close()
fig, (ax,) = az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
#ax.vlines(true_theta_pi_r, ylims)
plt.savefig("./plots/posterior_plots/plot_forest_theta_pi_r.png")
#plt.show()
plt.close()

try:
    scores = geweke(idata, first=0.1, last=0.5, intervals=20)
    pm.Matplot.geweke_plot(scores) 
except:
    print("scores = az.geweke(idata, first=0.1, last=0.5, intervals=20) did not work")

try:
    scores = az.geweke(idata, first=0.1, last=0.5, intervals=20)
    pm.Matplot.geweke_plot(scores) 
except:
    print("scores = az.geweke(idata, first=0.1, last=0.5, intervals=20) did not work")

#pm.Matplot.geweke_plot(scores, name='geweke', format='png', suffix='-diagnostic', path='./', fontmap = {1:10, 2:8, 3:6, 4:5, 5:4}, verbose=1)

# Remember to handle missingness and standardize: 
# Standardize the features
#X -= X.mean()
#X /= X.std()
#N_patients, P = X.shape
