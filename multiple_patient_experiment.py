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
P = 6 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
X_mean = np.repeat(0,P)
X_std = np.repeat(0.5,P)
X = np.random.normal(X_mean, X_std, size=(N_patients,P))
X = pd.DataFrame(X, columns = ['Covariate_1','Covariate_2','Covariate_3', 'Covariate_4','Covariate_5','Covariate_6'])
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
true_beta_rho_s = np.array([0.8, 0.9, 0.0, 0.0, 0.0, 0.0])
true_beta_rho_r = np.array([0.7, 1.0, 0.0, 0.0, 0.0, 0.0])
true_beta_pi_r = np.array([0.0, 1.1, 0.0, 0.0, 0.0, 0.0])

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

#Y = np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()])
#t = np.array([patient.measurement_times for _, patient in patient_dictionary.items()])
#yi0 = np.array([[patient.Mprotein_values[0]] for _, patient in patient_dictionary.items()])
Y = np.transpose(np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()]))
t = np.transpose(np.array([patient.measurement_times for _, patient in patient_dictionary.items()]))
yi0 = np.array([patient.Mprotein_values[0] for _, patient in patient_dictionary.items()])
#print("Y:\n", Y)
#print("t:\n", t)
#print("yi0:\n", yi0)
X_not_transformed = X.copy()
print(X_not_transformed.columns.values)
X = X.T
print("Done generating data")
##############################
import aesara.tensor as at
with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as multiple_patients_model:
    # Observation noise (std)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # alpha
    alpha = pm.Normal("alpha",  mu=true_alpha,  sigma=1, shape=3)
    # beta (with horseshoe priors)
    # Global shrinkage prior
    tau_rho_s = pm.HalfStudentT("tau_rho_s", 2, P0 / (P - P0) * sigma / np.sqrt(N_patients))
    tau_rho_r = pm.HalfStudentT("tau_rho_r", 2, P0 / (P - P0) * sigma / np.sqrt(N_patients))
    tau_pi_r = pm.HalfStudentT("tau_pi_r", 2, P0 / (P - P0) * sigma / np.sqrt(N_patients))
    # Local shrinkage prior
    lam_rho_s = pm.HalfStudentT("lam_rho_s", 2, dims="predictors")
    lam_rho_r = pm.HalfStudentT("lam_rho_r", 2, dims="predictors")
    lam_pi_r = pm.HalfStudentT("lam_pi_r", 2, dims="predictors")
    c2_rho_s = pm.InverseGamma("c2_rho_s", 1, 0.1)
    c2_rho_r = pm.InverseGamma("c2_rho_r", 1, 0.1)
    c2_pi_r = pm.InverseGamma("c2_pi_r", 1, 0.1)
    z_rho_s = pm.Normal("z_rho_s", 0.0, 1.0, dims="predictors")
    z_rho_r = pm.Normal("z_rho_r", 0.0, 1.0, dims="predictors")
    z_pi_r = pm.Normal("z_pi_r", 0.0, 1.0, dims="predictors")
    # Shrunken coefficients
    beta_rho_s = pm.Deterministic("beta_rho_s", z_rho_s * tau_rho_s * lam_rho_s * at.sqrt(c2_rho_s / (c2_rho_s + tau_rho_s**2 * lam_rho_s**2)), dims="predictors")
    beta_rho_r = pm.Deterministic("beta_rho_r", z_rho_r * tau_rho_r * lam_rho_r * at.sqrt(c2_rho_r / (c2_rho_r + tau_rho_r**2 * lam_rho_r**2)), dims="predictors")
    beta_pi_r = pm.Deterministic("beta_pi_r", z_pi_r * tau_pi_r * lam_pi_r * at.sqrt(c2_pi_r / (c2_pi_r + tau_pi_r**2 * lam_pi_r**2)), dims="predictors")

    ##tau_rho_s = pm.HalfStudentT("tau_rho_s", 2, P0 / (P - P0) * sigma / np.sqrt(N_patients))
    ##tau_rho_r = pm.HalfStudentT("tau_rho_r", 2, P0 / (P - P0) * sigma / np.sqrt(N_patients))
    ##tau_pi_r = pm.HalfStudentT("tau_pi_r", 2, P0 / (P - P0) * sigma / np.sqrt(N_patients))
    ### Local shrinkage prior
    ##lam_rho_s = pm.HalfStudentT("lam_rho_s", 2, shape=(P,1))
    ##lam_rho_r = pm.HalfStudentT("lam_rho_r", 2, shape=(P,1))
    ##lam_pi_r = pm.HalfStudentT("lam_pi_r", 2, shape=(P,1))
    ##c2_rho_s = pm.InverseGamma("c2_rho_s", 1, 0.1)
    ##c2_rho_r = pm.InverseGamma("c2_rho_r", 1, 0.1)
    ##c2_pi_r = pm.InverseGamma("c2_pi_r", 1, 0.1)
    ##z_rho_s = pm.Normal("z_rho_s", 0.0, 1.0, shape=(P,1))
    ##z_rho_r = pm.Normal("z_rho_r", 0.0, 1.0, shape=(P,1))
    ##z_pi_r = pm.Normal("z_pi_r", 0.0, 1.0, shape=(P,1))
    ### Shrunken coefficients
    ##beta_rho_s = pm.Deterministic("beta_rho_s", z_rho_s * tau_rho_s * lam_rho_s * at.sqrt(c2_rho_s / (c2_rho_s + tau_rho_s**2 * lam_rho_s**2))) #, dims="predictors")
    ##beta_rho_r = pm.Deterministic("beta_rho_r", z_rho_r * tau_rho_r * lam_rho_r * at.sqrt(c2_rho_r / (c2_rho_r + tau_rho_r**2 * lam_rho_r**2))) #, dims="predictors")
    ##beta_pi_r = pm.Deterministic("beta_pi_r", z_pi_r * tau_pi_r * lam_pi_r * at.sqrt(c2_pi_r / (c2_pi_r + tau_pi_r**2 * lam_pi_r**2))) #, dims="predictors")
    #beta_rho_s = pm.Normal("beta_rho_s", mu=0, sigma=1, shape=(P,1))
    #beta_rho_r = pm.Normal("beta_rho_r", mu=0, sigma=1, shape=(P,1))
    #beta_pi_r = pm.Normal("beta_pi_r", mu=0, sigma=1, shape=(P,1))

    # Latent variables theta
    #theta_rho_s = alpha[0] + pm.math.dot(X, beta_rho_s) # Deterministically determined by x, beta, alpha
    #theta_rho_r = alpha[1] + pm.math.dot(X, beta_rho_r) # Deterministically determined by x, beta, alpha
    #theta_pi_r  = alpha[2] + pm.math.dot(X, beta_pi_r) # Deterministically determined by x, beta, alpha
    omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
    theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + at.dot(beta_rho_s, X), sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
    theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + at.dot(beta_rho_r, X), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
    theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + at.dot(beta_pi_r, X),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

    # Transformed latent variables 
    rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
    rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
    pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

    # psi: True M protein at time 0
    #psi = pm.Normal("psi", mu=psi_population, sigma=10, shape=(N_patients,1)) # Informative on population level
    psi = pm.Normal("psi", mu=yi0, sigma=sigma, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 

    # Observation model 
    mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma, observed=Y)
# Visualize model
import graphviz 
gv = pm.model_to_graphviz(multiple_patients_model)
gv.render(filename='./graph_of_model', format="png", view=True)
#gv = pm.model_graph.model_to_graphviz(model)
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

#print(az.summary(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), round_to=2)) #, stat_focus="median")
print(az.summary(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma'), round_to=2)) #, stat_focus="median")

az.plot_trace(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma'), combined=True)
plt.savefig("./plots/posterior_plots/plot_posterior_group_parameters.png")
plt.show()
plt.close()

az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
plt.savefig("./plots/posterior_plots/plot_posterior_individual_parameters.png")
plt.show()
plt.close()

# Test of exploration 
az.plot_energy(idata)
plt.savefig("./plots/posterior_plots/plot_energy.png")
plt.show()
plt.close()
# Plot of coefficients
az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_alpha.png")
#plt.show()
az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta_rho_s.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta_rho_r.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_beta_pi_r.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_s.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_rho_r.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/posterior_plots/plot_forest_theta_pi_r.png")
#plt.show()
plt.close()

# Remember to handle missingness and standardize: 
# Standardize the features
#X -= X.mean()
#X /= X.std()
#N_patients, P = X.shape
