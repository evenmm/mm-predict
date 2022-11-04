from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import aesara.tensor as at
from create_training_instance_dictionary_with_covariates import *
from feature_extraction import *
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
yi0 = np.array([elem[0] for elem in y])
yi0 = np.maximum(1e-3, yi0)
#print(times)
#print([elem[0] for elem in times])
#print(yi0)

Y = np.transpose(y)
t = np.transpose(times)
print("Average Y", np.mean(Y))
print("Average t", np.mean(t))
print("Average yi0:", np.mean(yi0))

X = feature_extraction(training_instance_dict)

#P_sim = 6
#X_mean = np.repeat(0,P_sim)
#X_std = np.repeat(0.5,P_sim)
#X = np.random.normal(X_mean, X_std, size=(N_patients,P_sim))
##print(X)
##print(X[:,0])
##print(np.transpose(np.array([patient.covariates for _, patient in patient_dictionary.items()]))[0])
#X[:,0] = np.transpose(np.array([patient.covariates for _, patient in patient_dictionary.items()]))[0] #np.array([patient.covariates for _, patient in patient_dictionary.items()])
##print(X)
##print(X[:,0])
# Remember to handle missingness
# Standardize the features
#X -= X.mean()
#X /= X.std()
_, P = X.shape
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
print("P:",P)
print("P0:",P0)
print("X:", X)
X = X
print(X.dtypes)
#X = pd.DataFrame(X, columns = ['t0'])
#X = pd.DataFrame(X, columns = ['t0','Covariate_2','Covariate_3', 'Covariate_4','Covariate_5','Covariate_6'])
X_not_transposed = X.copy()
print("Covariates:", X_not_transposed.columns.values)
X = X.T
print("Shapes:")
print("Y:", Y.shape)
print("t:", t.shape)
print("yi0:", yi0.shape)
print("X:", X.shape)
print("Done loading data")
##############################
# Find initial estimates of rho and pi: 
with pm.Model(coords={"predictors": X_not_transposed.columns.values}) as model_for_finding_initial_alpha_guess:
    # Observation noise (std)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # alpha
    alpha = pm.Normal("alpha",  mu=np.array([np.log(0.0001), np.log(0.0002), np.log(0.2/(1-0.2))]),  sigma=1, shape=3)

    # Latent variables theta
    omega = pm.HalfNormal("omega",  sigma=10, shape=3) # Patient variability in theta (std)
    theta_rho_s = pm.Normal("theta_rho_s", mu=alpha[0], sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
    theta_rho_r = pm.Normal("theta_rho_r", mu=alpha[1], sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
    theta_pi_r  = pm.Normal("theta_pi_r",  mu=alpha[2],  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

    # psi: True M protein at time 0
    # log_psi = pm.Normal("log_psi", mu=np.log(yi0) - np.log( (sigma**2)/(yi0**2) - 1), sigma=np.log( (sigma**2)/(yi0**2) - 1), shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
    #log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=1, shape=N_patients)
    #psi = pm.Deterministic("psi", np.exp(log_psi))
    # This didnæt work either. None of these work. 
    psi = pm.Normal("psi", mu=yi0, sigma=sigma, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
    
    # Transformed latent variables 
    rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
    rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
    pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

    # Observation model 
    mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma, observed=Y)
with model_for_finding_initial_alpha_guess:
    idata = pm.sample() # draw 1000 posterior samples

#print(idata.posterior["alpha"])
alpha_mode_samples = scipy.stats.mode(idata.posterior["alpha"]).mode
alpha_guess = np.mean(alpha_mode_samples, axis=(0,1))
print("alpha_guess:", alpha_guess)
print("Population means:", alpha_guess)

rho_s_guess = -np.exp(alpha_guess[0])
rho_r_guess = np.exp(alpha_guess[1])
pi_r_guess  = 1/(1+np.exp(-alpha_guess[2]))
print("rho_s_guess:", rho_s_guess)
print("rho_r_guess:", rho_r_guess)
print("pi_r_guess:", pi_r_guess)

az.plot_trace(idata, var_names=('alpha', 'omega', 'sigma'), combined=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_alphaguess_fit__posterior_group_parameters.png")
#plt.show()
plt.close()

az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_alphaguess_fit__posterior_individual_parameters.png")
#plt.show()
plt.close()

########## Actual fitting of the entire model
with pm.Model(coords={"predictors": X_not_transposed.columns.values}) as multiple_patients_model:
    # Observation noise (std)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # alpha
    #alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)
    #alpha = pm.Normal("alpha",  mu=np.array([np.log(rho_s_guess[0]), np.log(rho_r_guess[1]), np.log(pi_r_guess[2]/(1-pi_r_guess[2]))]),  sigma=1, shape=3)
    alpha = pm.Normal("alpha",  mu=np.array([alpha_guess[0], alpha_guess[1], alpha_guess[2]]),  sigma=1, shape=3)
    #alpha = pm.Normal("alpha",  mu=np.array([-6.6, -7.5, -1.3]),  sigma=1, shape=3)

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

    # Latent variables theta
    omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
    theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + at.dot(beta_rho_s, X), sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
    theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + at.dot(beta_rho_r, X), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
    theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + at.dot(beta_pi_r, X),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

    # psi: True M protein at time 0
    #psi = pm.Normal("psi", mu=yi0, sigma=sigma, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
    #log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=1, shape=N_patients)
    #psi = pm.Deterministic("psi", np.exp(log_psi))
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
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_prior_samples.png")
#plt.show()
plt.close()
# Sample from posterior:
with multiple_patients_model:   
    # draw 1000 posterior samples
    #idata = pm.sample()
    idata = pm.sample(4000, tune=8000, random_seed=42, target_accept=0.99)

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

az.plot_trace(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma'), combined=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_posterior_group_parameters.png")
plt.show()
plt.close()

az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_posterior_individual_parameters.png")
#plt.show()
plt.close()

# Test of exploration 
az.plot_energy(idata)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_energy.png")
plt.show()
plt.close()
# Plot of coefficients
az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_alpha.png")
#plt.show()
az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_beta_rho_s.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_beta_rho_r.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_beta_pi_r.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_theta_rho_s.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_theta_rho_r.png")
#plt.show()
plt.close()
az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
plt.savefig("./plots/COMMPASS_posterior/COMMPASS_plot_forest_theta_pi_r.png")
#plt.show()
plt.close()
