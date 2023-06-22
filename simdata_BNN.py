from utilities import *
from BNN_model import *
from partial_BNN_model import *

# Initialize random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
#SAVEDIR = "/data/evenmm/plots/"
SAVEDIR = "./plots/Bayesian_estimates_simdata_BNN/"
#SAVEDIR = "./"

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

model_name = "BNN"
N_patients = 150
N_patients_test = 30
psi_prior="lognormal"
WEIGHT_PRIOR = "Student_out" #"Horseshoe" # "Student_out" #"symmetry_fix" #"iso_normal" "Student_out"
net_list = ["pi", "rho_r"] #, "rho_s"] # Which variables should be predicted by a neural net
N_samples = 10
N_tuning = 10
n_chains = 4
advi_iterations = 100
N_iter_individual_map = 10
maxeval_map = 300
INFERENCE_MODE = "Full" #"Partial"
EMPIRICAL_BAYES = False
ADADELTA = True
target_accept = 0.99
CI_with_obs_noise = False
PLOT_RESISTANT = True
FUNNEL_REPARAMETRIZATION = False
MODEL_RANDOM_EFFECTS = True
DIFFERENT_LENGTHS = True
N_HIDDEN = 2
P = 5 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
true_omega = np.array([0.10, 0.05, 0.20])
test_seed = 23

M_number_of_measurements = 7
y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
true_omega_for_psi = 0.1

max_time = 1200 #3000 #1500
days_between_measurements = int(max_time/M_number_of_measurements)
measurement_times = days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])
name = "simdata_"+model_name+"_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_FUNNEL_"+str(FUNNEL_REPARAMETRIZATION)+"_RNDM_EFFECTS_"+str(RANDOM_EFFECTS)+"_WT_PRIOR_"+str(WEIGHT_PRIOR+"_N_HIDDN_"+str(N_HIDDEN))
print("Running "+name)

X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_3_pi_rho, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS, DIFFERENT_LENGTHS=DIFFERENT_LENGTHS)

# Visualize parameter dependancy on covariates 
plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r)
"""
# Individual fit for Empirical Bayes (and MAP initialization)
rho_s, rho_r, pi_r = estimate_individual_parameters(patient_dictionary, N_iter=N_iter_individual_map, plotting=True)
print()
print("mean rho_s:", np.mean(rho_s))
print("std rho_s:", np.std(rho_s), "\n")
print("mean rho_r:", np.mean(rho_r))
print("std rho_r:", np.std(rho_r), "\n")
print("mean pi_r:", np.mean(pi_r))
print("std pi_r:", np.std(pi_r), "\n")
"""
rho_s, rho_r, pi_r = [1], [2], [3]

# Use log transformed std and mena to create Empirical Bayes priors for alpha 1 2 3: 
if EMPIRICAL_BAYES:
    empirical_mean_alpha = np.array([np.mean(rho_s), np.mean(rho_r), np.mean(pi_r)])
else:
    empirical_mean_alpha = []

if INFERENCE_MODE == "Partial": # Partially stochastic inference
    print("------------------- PARTIALLY BAYESIAN INFERENCE -------------------")
    print("Finding MAP estimate for full model")
    # First get MAP estimate for the full model
    # Using scikit learn neural net and the MAP estimated parameters as labels 
    #from sklearn.neural_network import MLPRegressor
    #from sklearn.model_selection import train_test_split
    #sklearn_nn = np.empty(3, dtype=object)
    #for i, y in enumerate([rho_s, rho_r, pi_r]):
    #    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #    regr = MLPRegressor(random_state=1, max_iter=50000).fit(X_train, y_train)
    #    print("regr.predict(X_test[:2])", regr.predict(X_test[:2]))
    #    print("y_test[:2]", y_test[:2])
    #    print("regr.score(X_test, y_test)", regr.score(X_test, y_test))
    #    sklearn_nn[i] = MLPRegressor(hidden_layer_sizes=N_HIDDEN, activation="relu", solver="lbfgs", random_state=1, max_iter=50000).fit(X, y)
    #    print(sklearn_nn[i].coefs_[1].flatten())

    #map_estimate = {"weights_out_rho_s": sklearn_nn[0].coefs_[1].flatten(), "weights_out_rho_r": sklearn_nn[1].coefs_[1].flatten(), "weights_out_pi_r": sklearn_nn[2].coefs_[1].flatten()}
    map_estimate = {"weights_out_rho_s": np.array([-0.09827992, 0.2271479]), "weights_out_rho_r": np.array([0.00185181, -0.00504315]), "weights_out_pi_r": np.array([0.27789275, 0.57438035])}

    # Using scipy.optimize.minimize to get a MAP estimate of the full model:
    #prelim_full_model = BNN_model(X, patient_dictionary, name, psi_prior=psi_prior, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, WEIGHT_PRIOR=WEIGHT_PRIOR, n_hidden=N_HIDDEN, net_list=net_list)
    #with prelim_full_model:
    #    map_estimate = pm.find_MAP(maxeval=maxeval_map, method="L-BFGS-B", tol=1e-10, options={"maxiter": maxeval_map, "gtol": 1e-10, "ftol": 1e-10, "eps": 1e-10})

    # Then fix some weights to the MAP estimate and do inference on the remaining weights
    neural_net_model = partial_BNN_model(map_estimate, X, patient_dictionary, name, psi_prior=psi_prior, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, WEIGHT_PRIOR=WEIGHT_PRIOR, n_hidden=N_HIDDEN, net_list=net_list, empirical_mean_alpha=empirical_mean_alpha)
else: 
    map_estimate = {"weights_out_rho_s": np.array([-0.09827992, 0.2271479]), "weights_out_rho_r": np.array([0.00185181, -0.00504315]), "weights_out_pi_r": np.array([0.27789275, 0.57438035])}
    print("------------------- FULL BAYESIAN INFERENCE -------------------")
    # Fully Bayesian, all weights are random variables
    neural_net_model = BNN_model(X, patient_dictionary, name, psi_prior=psi_prior, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, WEIGHT_PRIOR=WEIGHT_PRIOR, n_hidden=N_HIDDEN, net_list=net_list, empirical_mean_alpha=empirical_mean_alpha)

# Draw samples from posterior, no matter which inference mode:
with neural_net_model:
    if ADADELTA: 
        print("------------------- ADADELTA ADVI INITIALIZATION -------------------")
        advi_iterations = advi_iterations
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
        sample = approx.sample(return_inferencedata=False, size=n_chains)
        start_dict = list(sample[i] for i in range(n_chains))    
        # essentially, this is what init='advi' does!!!
        step = pm.NUTS(scaling=scaling, is_cov=True)
        idata = pm.sample(draws=N_samples, tune=N_tuning, step=step, chains=n_chains, initvals=start_dict) #, random_seed=42, target_accept=target_accept)
    else: 
        idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=60000, random_seed=42, target_accept=target_accept)

print("Done sampling")

picklefile = open(SAVEDIR+name+'_idata', 'wb')
pickle.dump(idata, picklefile)
picklefile.close()

# Generate test patients
X_test, patient_dictionary_test, parameter_dictionary_test, expected_theta_1_test, true_theta_rho_s_test, true_rho_s_test, expected_theta_2_test, true_theta_rho_r_test, true_rho_r_test, expected_theta_3_test, true_theta_pi_r_test, true_pi_r_test = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_test, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=test_seed, RANDOM_EFFECTS=RANDOM_EFFECTS_TEST)
print("Done generating test patients")

#plot_all_credible_intervals(idata, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name, y_resolution, model_name=model_name, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=True, net_list=net_list, PARALLELLIZE=True, INFERENCE_MODE=INFERENCE_MODE, MAP_weights=map_estimate)
#print("Finished!")
#plot_posterior_traces(idata, SAVEDIR, name, psi_prior, model_name=model_name, net_list=net_list, INFERENCE_MODE="Full")
#quasi_geweke_test(idata, model_name=model_name, first=0.1, last=0.5)

# At how many days to we want to classify people into recurrence / not recurrence: 
# Here we make sure that all patients have observations at that time by taking the latest time where every patient has an observation
evaluation_time = measurement_times[:3][-1] + 1
print(evaluation_time)
true_pfs = get_true_pfs(patient_dictionary_test)
recurrence_or_not, p_recurrence, predicted_PFS, point_predicted_PFS = pfs_auc(evaluation_time, patient_dictionary_test, N_patients_test, idata, X_test, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, SAVEDIR, name, INFERENCE_MODE, map_estimate, net_list)

print("True PFS", true_pfs)
print("Average PFS among actual PFS", np.mean(true_pfs[true_pfs>=0]))
print("Std PFS among actual PFS", np.std(true_pfs[true_pfs>=0]))
print("Median PFS among actual PFS", np.median(true_pfs[true_pfs>=0]))
print("Max PFS among actual PFS", np.max(true_pfs[true_pfs>=0]))
print("Min PFS among actual PFS", np.min(true_pfs[true_pfs>=0]))

print("\npoint_predicted_PFS", point_predicted_PFS)
print("Average PFS among actual PFS", np.mean(point_predicted_PFS[point_predicted_PFS>=0]))
print("Std PFS among actual PFS", np.std(point_predicted_PFS[point_predicted_PFS>=0]))
print("Median PFS among actual PFS", np.median(point_predicted_PFS[point_predicted_PFS>=0]))
print("Max PFS among actual PFS", np.max(point_predicted_PFS[point_predicted_PFS>=0]))
print("Min PFS among actual PFS", np.min(point_predicted_PFS[point_predicted_PFS>=0]))
# RMSE of predicted PFS:
print("\nRMSE of predicted PFS:", np.sqrt(np.mean((true_pfs - point_predicted_PFS)**2)))
print("Mean Absolute error of predicted PFS:", np.mean(np.abs(true_pfs - point_predicted_PFS)))

