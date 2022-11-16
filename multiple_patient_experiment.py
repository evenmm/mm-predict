from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import aesara.tensor as at
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
##############################
# Generate data
true_sigma = 1
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

days_between_measurements = 300
number_of_measurements = 4
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements-1, number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s), (N_patients,1))
expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients,1))
expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients,1))

# Patient specific noise / deviation from X effects
true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])
print("true_theta_rho_s[0:5]:\n", true_theta_rho_s[0:5])
print("true_theta_rho_r[0:5]:\n", true_theta_rho_r[0:5])
print("true_theta_pi_r[0:5]:\n", true_theta_pi_r[0:5])

true_rho_s = - np.exp(true_theta_rho_s)
true_rho_r = np.exp(true_theta_rho_r)
true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
true_psi = np.exp(np.random.normal(np.log(psi_population),0.1,size=N_patients))
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
with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as multiple_patients_model:
    # Observation noise (std)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # alpha
    alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)

    # beta (with horseshoe priors):
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

    # Latent variables theta
    omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
    theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + at.dot(beta_rho_s, X), sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
    theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + at.dot(beta_rho_r, X), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
    theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + at.dot(beta_pi_r, X),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

    # psi: True M protein at time 0
    # Exact but does not work: 
    #log_psi = pm.Normal("log_psi", mu=np.log(yi0) - np.log( (sigma**2)/(yi0**2) - 1), sigma=np.log( (sigma**2)/(yi0**2) - 1), shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
    #psi = pm.Deterministic("psi", np.exp(log_psi))
    # Bad:
    #log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=1, shape=N_patients)
    #psi = pm.Deterministic("psi", np.exp(log_psi))
    # Worse: 
    #psi = pm.HalfNormal("psi", sigma=10, shape=N_patients)
    # This wrong because normal, but still the best on our case: 
    psi = pm.Normal("psi", mu=yi0, sigma=sigma, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 

    # Transformed latent variables 
    rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
    rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
    pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

    # Observation model 
    mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma, observed=Y)
# Visualize model
import graphviz 
gv = pm.model_to_graphviz(multiple_patients_model)
gv.render(filename='./graph_of_model', format="png", view=False)
# Sample from prior:
with multiple_patients_model:
    prior_samples = pm.sample_prior_predictive(200)
thresholded_Y_true = np.ravel(Y)
thresholded_Y_true[thresholded_Y_true > 200] = 170
thresholded_Y_sampl = np.ravel(prior_samples.prior_predictive["Y_obs"])
thresholded_Y_sampl[thresholded_Y_sampl > 200] = 170
az.plot_dist(
    #np.log(thresholded_Y_true),
    thresholded_Y_true,
    color="C1",
    label="observed",
    #backend_kwargs={"set_xlim":"([-10,30])"}
)
az.plot_dist(
    #np.log(thresholded_Y_sampl),
    thresholded_Y_sampl,
    label="simulated",
    #backend_kwargs={"set_xlim":"([-10,30])"}
)
plt.title("Samples from prior compared to observations")
plt.xlabel("Y (M protein)")
plt.ylabel("Frequency")
plt.savefig("./plots/posterior_plots/plot_prior_samples.png")
#plt.show()
plt.close()
# Sample from posterior:
with multiple_patients_model:
    # draw 1000 posterior samples
    idata = pm.sample(1000, tune=1000, random_seed=42, target_accept=0.99)

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
