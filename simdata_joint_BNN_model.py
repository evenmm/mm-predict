from utilities import *
from joint_BNN_model import *

# Initialize random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
SAVEDIR = "/data/evenmm/plots/"
#SAVEDIR = "./plots/Bayesian_estimates_simdata_joint_BNN/"

script_index = int(sys.argv[1]) 

# Settings
if int(script_index % 3) == 0:
    true_sigma_obs = 0
elif int(script_index % 3) == 1:
    true_sigma_obs = 1
elif int(script_index % 3) == 2:
    true_sigma_obs = 2.5

if script_index >= 3:
    RANDOM_EFFECTS = True
else: 
    RANDOM_EFFECTS = False

RANDOM_EFFECTS_TEST = False

model_name = "joint_BNN"
N_patients = 150
psi_prior="lognormal"
WEIGHT_PRIOR = "Student_out" #"Horseshoe" # "Student_out" #"symmetry_fix" #"iso_normal" "Student_out"
N_samples = 10_000
N_tuning = 10_000
ADADELTA = True
target_accept = 0.99
PLOT_RESISTANT = True
CI_with_obs_noise = False
FUNNEL_REPARAMETRIZATION = False
MODEL_RANDOM_EFFECTS = True
N_HIDDEN = 2
P = 5 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
true_omega = np.array([0.10, 0.05, 0.20])

M_number_of_measurements = 5
y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
true_omega_for_psi = 0.1

max_time = 1200 #3000 #1500
days_between_measurements = int(max_time/M_number_of_measurements)
measurement_times = days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])
name = "simdata_"+model_name+"_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_FUNNEL_"+str(FUNNEL_REPARAMETRIZATION)+"_RNDM_EFFECTS_"+str(RANDOM_EFFECTS)+"_WT_PRIOR_"+str(WEIGHT_PRIOR+"_N_HIDDN_"+str(N_HIDDEN))
print("Running "+name)

X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS)

# Visualize parameter dependancy on covariates 
#plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s)

# Sample from full model
neural_net_model = joint_BNN_model(X, patient_dictionary, name, psi_prior=psi_prior, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, WEIGHT_PRIOR=WEIGHT_PRIOR, n_hidden=N_HIDDEN)
# Draw samples from posterior:
with neural_net_model:
    if ADADELTA: 
        print("------------------- ADADELTA INITIALIZATION -------------------")
        advi_iterations = 100_000
        advi = pm.ADVI()
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  # callable that returns mean
            std=advi.approx.std.eval,  # callable that returns std
        )
        approx = advi.fit(advi_iterations, obj_optimizer=pm.adadelta(), obj_n_mc=50, callbacks=[tracker])
        #approx = advi.fit(advi_iterations, obj_optimizer=pm.adagrad(), obj_n_mc=5, callbacks=[tracker])

        # Plot ELBO and trace
        fig = plt.figure(figsize=(16, 9))
        mu_ax = fig.add_subplot(221)
        std_ax = fig.add_subplot(222)
        hist_ax = fig.add_subplot(212)
        mu_ax.plot(tracker["mean"])
        mu_ax.set_title("Mean track")
        std_ax.plot(tracker["std"])
        std_ax.set_title("Std track")
        hist_ax.plot(advi.hist)
        hist_ax.set_title("Negative ELBO track")
        hist_ax.set_yscale("log")
        plt.savefig(SAVEDIR+"0_elbo_and_trace_"+name+".pdf", dpi=300)
        #plt.show()
        plt.close()
        
        print("-------------------SAMPLING-------------------")
        # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
        scaling = approx.cov.eval()
        n_chains = 4
        sample = approx.sample(return_inferencedata=False, size=n_chains)
        start_dict = list(sample[i] for i in range(n_chains))    
        print(start_dict)
        # essentially, this is what init='advi' does!!!
        step = pm.NUTS(scaling=scaling, is_cov=True)
        idata = pm.sample(draws=N_samples, tune=N_tuning, step=step, chains=n_chains, initvals=start_dict) #, random_seed=42, target_accept=target_accept)
    else: 
        idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=60000, random_seed=42, target_accept=target_accept)

print("Done sampling")

picklefile = open(SAVEDIR+name+'_idata', 'wb')
pickle.dump(idata, picklefile)
picklefile.close()

#quasi_geweke_test(idata, model_name="BNN", first=0.1, last=0.5)

print("Plotting posterior/trace plots")
plot_posterior_traces(idata, SAVEDIR, name, psi_prior, model_name=model_name)

# Generate test patients
N_patients_test = 20
test_seed = 23
X_test, patient_dictionary_test, parameter_dictionary_test, expected_theta_1_test, true_theta_rho_s_test, true_rho_s_test = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_test, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=test_seed, RANDOM_EFFECTS=RANDOM_EFFECTS_TEST)
print("Done generating test patients")

plot_all_credible_intervals(idata, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name, y_resolution, model_name=model_name, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=PLOT_RESISTANT)
