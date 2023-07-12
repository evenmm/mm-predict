from utilities import *
from linear_model import *
from individual_model import *
import sklearn.metrics as metrics
import gc 
# Provided partial M protein from test patients
# Generate data using a linear model without interactions 
# Fit individual and linear models
# Compare AUC

# Initialize random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
#SAVEDIR = "/data/evenmm/plots/"
SAVEDIR = "./plots/Bayesian_estimates_simdata_comparison/"
#SAVEDIR = "./"

# Settings
script_index = int(sys.argv[1]) 
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

N_patients = 150
test_size = 0.4
psi_prior="lognormal"
N_samples = 1000
N_tuning = 1000
ADADELTA = True
advi_iterations = 15_000
n_init_advi = 6_000
n_chains = 4
CORES = 4
FUNNEL_REPARAMETRIZATION = False
MODEL_RANDOM_EFFECTS = True
target_accept = 0.99
CI_with_obs_noise = False
PLOT_RESISTANT = False
N_HIDDEN = 2
P = 5 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
true_omega = np.array([0.10, 0.05, 0.20])

y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
true_omega_for_psi = 0.1

max_time = 1 + 25*28
days_between_measurements = 28 #int(max_time/M_number_of_measurements)
#measurement_times = 1 + days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
measurement_times = np.array(range(1,max_time+days_between_measurements,days_between_measurements))
M_number_of_measurements = len(measurement_times) #int(max_time/days_between_measurements) #7
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

# Generate simulated patients 
# Put a USUBJID row in X with USUBJID=True
X, patient_dictionary_complete, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS, USUBJID=True)

# Split into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=test_size, random_state=RANDOM_SEED)
# Reset the index of X_train and X_test
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
patient_dictionary = get_ii_indexed_subset_dict(X_train, patient_dictionary_complete)
patient_dictionary_test = get_ii_indexed_subset_dict(X_test, patient_dictionary_complete)
X_train = X_train.copy().drop(columns=["USUBJID"])
X_test = X_test.copy().drop(columns=["USUBJID"])

# Create X_test_inference with partial M protein only 
N_patients_train, P = X_train.shape
assert len(patient_dictionary) == N_patients_train, "len(patient_dictionary)"+str(len(patient_dictionary))
N_patients_test, P = X_test.shape
assert len(patient_dictionary_test) == N_patients_test, "len(patient_dictionary_test)"+str(len(patient_dictionary_test))
assert X_train.shape[1] == X_test.shape[1] # P
X_full = pd.concat([X_train, X_test])
patient_dictionary_full = deepcopy(patient_dictionary)
for ii in range(len(patient_dictionary_test)):
    patient_dictionary_full[ii+N_patients_train] = deepcopy(patient_dictionary_test[ii])
# 2+3 Full inference for each clip time with their own "fit" dictionary
pred_window_length = 6*28
pred_window_starts = range(1+12*28, 1+42*28, pred_window_length)
CLIP_MPROTEIN_TIME = 180 #for CLIP_MPROTEIN_TIME in [360, 180, 90]:
end_of_prediction_horizon = CLIP_MPROTEIN_TIME + pred_window_length
patient_dictionary_fit = deepcopy(patient_dictionary)
for ii in range(len(patient_dictionary_test)):
    clip_patient = deepcopy(patient_dictionary_test[ii])
    clip_patient.Mprotein_values = clip_patient.Mprotein_values[clip_patient.measurement_times <= CLIP_MPROTEIN_TIME]
    clip_patient.measurement_times = clip_patient.measurement_times[clip_patient.measurement_times <= CLIP_MPROTEIN_TIME]
    patient_dictionary_fit[ii+N_patients_train] = clip_patient
assert len(patient_dictionary_full) == len(patient_dictionary_fit)
assert X_full.shape[0] == len(patient_dictionary_fit)
name_lin = "simdata_partial_Mprot_lin_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_CLIP_"+str(CLIP_MPROTEIN_TIME)+"_win_"+str(pred_window_length)
name_ind = "simdata_partial_Mprot_ind_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_CLIP_"+str(CLIP_MPROTEIN_TIME)+"_win_"+str(pred_window_length)
# Visualize parameter dependancy on covariates 
#plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r)
#plot_parameter_dependency_on_covariates(SAVEDIR, name_lin, X, expected_theta_1, true_theta_rho_s, true_rho_s)
ind_model = individual_model(patient_dictionary_fit, name_ind, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION)
lin_model = linear_model(X_full, patient_dictionary_fit, name_lin, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION)
for model, name in [(ind_model, name_ind), (lin_model, name_lin)]:
    try:
        print("Loading idata for " + name)
        picklefile = open(SAVEDIR+name+"_idata_pickle", "rb")
        idata = pickle.load(picklefile)
        picklefile.close()
    except:
        print("Sampling idata for " + name)
        picklefile = open(SAVEDIR+name+"_idata_pickle", "wb")
        with model:
            if ADADELTA:
                print("------------------- INDEPENDENT ADVI -------------------")
                advi = pm.ADVI()
                tracker = pm.callbacks.Tracker(
                    mean=advi.approx.mean.eval,  # callable that returns mean
                    std=advi.approx.std.eval,  # callable that returns std
                )
                approx = advi.fit(advi_iterations, obj_optimizer=pm.adagrad_window(), obj_n_mc=25, callbacks=[tracker], total_grad_norm_constraint=10_000.)

                print("-------------------SAMPLING-------------------")
                # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
                scaling = approx.cov.eval()
                sample = approx.sample(return_inferencedata=False, size=n_chains)
                start_dict = list(sample[i] for i in range(n_chains))    
                # essentially, this is what init='advi' does
                step = pm.NUTS(scaling=scaling, is_cov=True)
                idata = pm.sample(draws=N_samples, tune=N_tuning, step=step, initvals=start_dict, chains=n_chains , cores=CORES)
            else:
                idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=60000, random_seed=42, target_accept=target_accept, chains=n_chains, cores=CORES)
        print("Done sampling")
        pickle.dump(idata, picklefile)
        picklefile.close()
        dictfile = open(SAVEDIR+name+"_patient_dictionary", "wb")
        pickle.dump(patient_dictionary_fit, dictfile)
        dictfile.close()
        np.savetxt(SAVEDIR+name+"_patient_dictionary"+".csv", [patient.name for _, patient in patient_dictionary_fit.items()], fmt="%s")
    # 4 predictive plots for test, fit plots for train
    #if True:
    try:
        picklefile = open(SAVEDIR+name+"_p_progression", "rb")
        if name == name_ind:
            p_progression = pickle.load(picklefile)
        else:
            p_progression_LIN = pickle.load(picklefile)
        picklefile.close()
        print("Loaded p_progression")
    except:
        print("Getting p_progression without load")
        if name == name_ind:
            p_progression = predict_PFS_new(idata, patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon)
        else:
            p_progression_LIN = predict_PFS_new(idata, patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon)
        print("p_progression   ", p_progression)
        a_file = open(SAVEDIR+name+"_p_progression", "wb")
        if name == name_ind:
            pickle.dump(p_progression, a_file)
        else:
            pickle.dump(p_progression_LIN, a_file)
        a_file.close()
    PLOTTING = False
    if PLOTTING:
        plot_fit_and_predictions(idata, patient_dictionary_full, N_patients_train, SAVEDIR, name, y_resolution, CLIP_MPROTEIN_TIME, CI_with_obs_noise=False, PLOT_RESISTANT=False)
    del idata
    gc.collect()

# Velocity model
p_progression_velo = predict_PFS_velocity_model(patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon)
print("p_progression_velo", p_progression_velo)

# 5 Calculate predicted chance of PFS and plot ROC 
true_pfs = get_true_pfs_new(patient_dictionary_test, CLIP_MPROTEIN_TIME)
print("true_pfs", true_pfs)
# ROC Prediction interval: From CLIP to 6 months after
# SUBSET patients:
try:
    picklefile = open(SAVEDIR+name+"_binary_progress_or_not", "rb")
    binary_progress_or_not = pickle.load(picklefile)
    picklefile.close()
    picklefile = open(SAVEDIR+name+"_new_p_progression", "rb")
    new_p_progression = pickle.load(picklefile)
    picklefile.close()
    picklefile = open(SAVEDIR+name+"_new_p_progression_LIN", "rb")
    new_p_progression_LIN = pickle.load(picklefile)
    picklefile.close()
    picklefile = open(SAVEDIR+name+"_new_p_progression_velo", "rb")
    new_p_progression_velo = pickle.load(picklefile)
    picklefile.close()
    print("Loaded subset progressions new_p")
except:
    print("Getting subset progressions new_p without load")
    patient_dictionary_test_progression = {}
    ii_new = 0
    new_true_pfs = []
    new_p_progression = []
    new_p_progression_LIN = []
    new_p_progression_velo = []
    for ii, patient in patient_dictionary_test.items():
        # Check if any measurements in prediction interval
        any_measurements = np.any(np.logical_and(patient.measurement_times > CLIP_MPROTEIN_TIME, patient.measurement_times <= end_of_prediction_horizon))
        # and that the true pfs did not happen before CLIP. -1 (no progression) is fine
        not_already_progressed = true_pfs[ii] < 0 or true_pfs[ii] > CLIP_MPROTEIN_TIME
        if any_measurements and not_already_progressed:
            patient_dictionary_test_progression[ii_new] = deepcopy(patient)
            ii_new = ii_new + 1
            new_true_pfs.append(true_pfs[ii])
            new_p_progression_velo.append(p_progression_velo[ii])
            new_p_progression.append(p_progression[ii])
            new_p_progression_LIN.append(p_progression_LIN[ii])
    print("N_patients_test originally", N_patients_test)
    print("len(patient_dictionary_test_progression)", len(patient_dictionary_test_progression))
    binary_progress_or_not = [1 if x > CLIP_MPROTEIN_TIME and x <= end_of_prediction_horizon else 0 for x in new_true_pfs]
    print(pred_window_length, "day window, percentage progressing is", sum(binary_progress_or_not) / len(binary_progress_or_not), "All:", len(binary_progress_or_not), "Progressors", sum(binary_progress_or_not))
    a_file = open(SAVEDIR+name+"_binary_progress_or_not", "wb")
    pickle.dump(binary_progress_or_not, a_file)
    a_file.close()
    a_file = open(SAVEDIR+name+"_new_p_progression", "wb")
    pickle.dump(new_p_progression, a_file)
    a_file.close()
    a_file = open(SAVEDIR+name+"_new_p_progression_LIN", "wb")
    pickle.dump(new_p_progression_LIN, a_file)
    a_file.close()
    a_file = open(SAVEDIR+name+"_new_p_progression_velo", "wb")
    pickle.dump(new_p_progression_velo, a_file)
    a_file.close()

# 5 AUC
fpr_velo, tpr_velo, threshold_velo = metrics.roc_curve(binary_progress_or_not, new_p_progression_velo) #(y_test, preds)
fpr_naive, tpr_naive, threshold_naive = metrics.roc_curve(binary_progress_or_not, new_p_progression) #(y_test, preds)
fpr_LIN, tpr_LIN, threshold_LIN = metrics.roc_curve(binary_progress_or_not, new_p_progression_LIN) #(y_test, preds)
roc_auc_velo = metrics.auc(fpr_velo, tpr_velo)
roc_auc_naive = metrics.auc(fpr_naive, tpr_naive)
roc_auc_LIN = metrics.auc(fpr_LIN, tpr_LIN)
print("threshold_naive:\n", threshold_naive)
print("fpr_naive:\n", fpr_naive)
print("tpr_naive:\n", tpr_naive)
print("roc_auc_naive:\n", roc_auc_naive)
plt.figure()
plt.grid(visible=True)
plt.title("ROC curve from day "+str(CLIP_MPROTEIN_TIME)+" to "+str(end_of_prediction_horizon))
plt.plot([0,1], [0,1], color='grey', linestyle='--', label='_nolegend_')
plt.plot(fpr_velo, tpr_velo, color=plt.cm.viridis(0.9), label = 'Velocity model (AUC = %0.2f)' % roc_auc_velo)
plt.plot(fpr_naive, tpr_naive, color=plt.cm.viridis(0.6), label = 'Individual model (AUC = %0.2f)' % roc_auc_naive)
plt.plot(fpr_LIN, tpr_LIN, color=plt.cm.viridis(0.3), label = 'Covariate model (AUC = %0.2f)' % roc_auc_LIN)
plt.legend(loc = 'lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(SAVEDIR+name+"_AUC_"+str(len(binary_progress_or_not))+"_test_patients_"+str(sum(binary_progress_or_not))+"_progressors.pdf", dpi=300)
#plt.show()
plt.close()

## AUPR 
proportion_progressions = sum(binary_progress_or_not) / len(binary_progress_or_not)
precision_velo, recall_velo, threshold_velo = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression_velo) #(y_test, preds)
precision_naive, recall_naive, threshold_naive = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression) #(y_test, preds)
precision_LIN, recall_LIN, threshold_LIN = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression_LIN) #(y_test, preds)
aupr_velo = metrics.average_precision_score(binary_progress_or_not, new_p_progression_velo)
aupr_naive = metrics.average_precision_score(binary_progress_or_not, new_p_progression)
aupr_LIN = metrics.average_precision_score(binary_progress_or_not, new_p_progression_LIN)
print("threshold_naive:\n", threshold_naive)
print("precision_naive:\n", precision_naive)
print("recall_naive:\n", recall_naive)
print("aupr_naive:\n", aupr_naive)
plt.figure()
plt.grid(visible=True)
plt.title("AUPR curve from day "+str(CLIP_MPROTEIN_TIME)+" to "+str(end_of_prediction_horizon))
plt.plot([0,1], [proportion_progressions, proportion_progressions], color='grey', linestyle='--', label='_nolegend_')
plt.plot(recall_velo, precision_velo, color=plt.cm.viridis(0.9), label = 'Velocity model (AUPR = %0.2f)' % aupr_velo)
plt.plot(recall_naive, precision_naive, color=plt.cm.viridis(0.6), label = 'Individual model (AUPR = %0.2f)' % aupr_naive)
plt.plot(recall_LIN, precision_LIN, color=plt.cm.viridis(0.3), label = 'Covariate model (AUPR = %0.2f)' % aupr_LIN)
plt.legend(loc = 'lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Precision')
plt.xlabel('Recall (True positive rate)')
plt.savefig(SAVEDIR+name+"_AUPR_"+str(len(binary_progress_or_not))+"_test_patients_"+str(sum(binary_progress_or_not))+"_progressors.pdf", dpi=300)
#plt.show()
plt.close()


"""
recurrence_or_not, p_recurrence = pfs_auc(evaluation_time, patient_dictionary_test, N_patients_test, name_ind, X_test, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, SAVEDIR, name_ind)
print("Finished!")

# Sample from full linear model
print("Running "+name_lin)
with lin_model:
    if ADADELTA: 
        print("------------------- ADADELTA INITIALIZATION -------------------")
        advi_iterations = 100_00
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
        plt.savefig(SAVEDIR+"0_elbo_and_trace_"+name_lin+".pdf", dpi=300)
        #plt.show()
        plt.close()

        print("-------------------SAMPLING-------------------")
        # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
        scaling = approx.cov.eval()
        n_chains = 4
        sample = approx.sample(return_inferencedata=False, size=n_chains)
        start_dict = list(sample[i] for i in range(n_chains))    
        # essentially, this is what init='advi' does!!!
        step = pm.NUTS(scaling=scaling, is_cov=True)
        idata_lin = pm.sample(draws=N_samples, tune=N_tuning, step=step, chains=n_chains, initvals=start_dict) #, random_seed=42, target_accept=target_accept)
    else: 
        idata_lin = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", chains=n_chains, n_init=60000, random_seed=42, target_accept=target_accept)
print("Done sampling linear model")
picklefile = open(SAVEDIR+'idata_lin_'+name_lin, 'wb')
pickle.dump(idata_lin, picklefile)
picklefile.close()
quasi_geweke_test(idata_lin, model_name=model_name, first=0.1, last=0.5)
plot_posterior_traces(idata_lin, SAVEDIR, name_lin, psi_prior, model_name=model_name)

plot_all_credible_intervals(name_lin, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name_lin, y_resolution, model_name=model_name, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=True, PARALLELLIZE=True)
# At how many days to we want to classify people into recurrence / not recurrence: 
# Here we make sure that all patients have observations at that time by taking the latest time where every patient has an observation
evaluation_time = measurement_times[:3][-1] + 1
print(evaluation_time)
recurrence_or_not, p_recurrence = pfs_auc(evaluation_time, patient_dictionary_test, N_patients_test, name_lin, X_test, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, SAVEDIR, name_lin)



with ind_model:
    if ADADELTA: 
        print("------------------- ADADELTA INITIALIZATION -------------------")
        advi_iterations = 100_00
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
        plt.savefig(SAVEDIR+"0_elbo_and_trace_"+name_ind+".pdf", dpi=300)
        #plt.show()
        plt.close()

        print("-------------------SAMPLING-------------------")
        # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
        scaling = approx.cov.eval()
        n_chains = 4
        sample = approx.sample(return_inferencedata=False, size=n_chains)
        start_dict = list(sample[i] for i in range(n_chains))    
        # essentially, this is what init='advi' does!!!
        step = pm.NUTS(scaling=scaling, is_cov=True)
        idata_ind = pm.sample(draws=N_samples, tune=N_tuning, step=step, chains=n_chains, initvals=start_dict) #, random_seed=42, target_accept=target_accept)
    else: 
        idata_ind = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", chains=n_chains, n_init=60000, random_seed=42, target_accept=target_accept)
print("Done sampling no covariate model")
picklefile = open(SAVEDIR+'idata_ind_'+name_ind, 'wb')
pickle.dump(idata_ind, picklefile)
picklefile.close()
quasi_geweke_test(idata_ind, model_name=model_name, first=0.1, last=0.5)
plot_posterior_traces(idata_ind, SAVEDIR, name_ind, psi_prior, model_name=model_name)

## Generate test patients
#N_patients_test = 50
#test_seed = 23
#X_test, patient_dictionary_test, parameter_dictionary_test, expected_theta_1_test, true_theta_rho_s_test, true_rho_s_test = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_test, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=test_seed, RANDOM_EFFECTS=RANDOM_EFFECTS_TEST)
#print("Done generating test patients")

#plot_all_credible_intervals(name_ind, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name_ind, y_resolution, model_name=model_name, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=True, PARALLELLIZE=True)
"""
