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
print("true_sigma_obs", true_sigma_obs)
print("RANDOM_EFFECTS", RANDOM_EFFECTS)
#RANDOM_EFFECTS_TEST = False # Not relevant when we have provided M protein

N_patients = 300
psi_prior="lognormal"
N_samples = 1000
N_tuning = 1000
ADADELTA = True
advi_iterations = 60_000
n_init_advi = 6_000
n_chains = 4
CORES = 4
FUNNEL_REPARAMETRIZATION = False
MODEL_RANDOM_EFFECTS = True
target_accept = 0.99
CI_with_obs_noise = False
PLOT_RESISTANT = True
P = 5 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
PLOTTING = True

#true_omega = np.array([0.5, 0.3, 0.5]) # Good without covariate effects
true_omega = np.array([0.5, 0.05, 0.05])
simulate_rho_r_dependancy_on_rho_s = False
#coef_rho_s_rho_r = 0.3 if simulate_rho_r_dependancy_on_rho_s else 0.0
coef_rho_s_rho_r = 0
# Positive correlation between rho_s and rho_r ON THE THETA SCALE. Higher theta_rho_s (faster decline) means higher theta_rho_r (faster relapse)
model_rho_r_dependancy_on_rho_s = simulate_rho_r_dependancy_on_rho_s
print("simulate_rho_r_dependancy_on_rho_s", simulate_rho_r_dependancy_on_rho_s)

max_time = 1 + 15*28 #25*28 in real data
days_between_measurements = 28
measurement_times = np.array(range(1,max_time+days_between_measurements,days_between_measurements))
M_number_of_measurements = len(measurement_times)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])
DIFFERENT_LENGTHS = False

# Generate simulated patients 
# Put a USUBJID row in X with USUBJID=True
# Removed. Uniform. But backwards compatibility true_omega_for_psi = 0 #0.3 or 0.2
X, patient_dictionary_complete, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_2_0, true_omega, 0, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS, USUBJID=True, simulate_rho_r_dependancy_on_rho_s=simulate_rho_r_dependancy_on_rho_s, coef_rho_s_rho_r=coef_rho_s_rho_r, DIFFERENT_LENGTHS=DIFFERENT_LENGTHS)

plothowitlooks = False
if plothowitlooks:
    for training_instance_id, params in parameter_dictionary.items():
        pat_name = "Patient " + str(training_instance_id)
        patient = patient_dictionary_complete[pat_name]
        plot_mprotein(patient, pat_name, SAVEDIR+pat_name, PLOT_PARAMETERS=True, parameters=params, PLOT_lightchains=False, plot_pfs=False, plot_KapLam=False)

# Clip time defined first of all, this is the outer loop. 
# Then we define test, train, fold etc within. 
pred_window_length = 6*28
pred_window_starts = range(1+10*28, max_time, 1*28)
for CLIP_MPROTEIN_TIME in pred_window_starts:
    end_of_prediction_horizon = CLIP_MPROTEIN_TIME + pred_window_length
    print("CLIP_MPROTEIN_TIME", CLIP_MPROTEIN_TIME)
    print("end_of_prediction_horizon", end_of_prediction_horizon)
    # Stratify based on relapse status
    true_pfs_complete_patient_dictionary = get_true_pfs_new(patient_dictionary_complete, time_scale=1, M_scale=1)
    print("True PFS\n", true_pfs_complete_patient_dictionary)
    print("Average PFS among actual PFS", np.mean(true_pfs_complete_patient_dictionary[true_pfs_complete_patient_dictionary>0]))
    print("Std PFS among actual PFS", np.std(true_pfs_complete_patient_dictionary[true_pfs_complete_patient_dictionary>0]))
    print("Median PFS among actual PFS", np.median(true_pfs_complete_patient_dictionary[true_pfs_complete_patient_dictionary>0]))
    print("Max PFS among actual PFS", np.max(true_pfs_complete_patient_dictionary[true_pfs_complete_patient_dictionary>0]))
    print("Min PFS among actual PFS", np.min(true_pfs_complete_patient_dictionary[true_pfs_complete_patient_dictionary>0]))
    # We don't subset patients by already relapsed here, or whether they have any measurements inside the interval. This is just for balance, not evaluation. 
    # Already relapsed are classified as relapse = 0 within period for purposes of stratification, because relapsers are fewer and hence more volatile
    relapse_label = [1 if x > CLIP_MPROTEIN_TIME and x <= end_of_prediction_horizon else 0 for x in true_pfs_complete_patient_dictionary]
    print(pred_window_length, "day window, percentage progressing is", sum(relapse_label) / len(relapse_label), "All:", len(relapse_label), "Progressors", sum(relapse_label))
    assert len(relapse_label) == X.shape[0]
    stored_fpr_velo = []
    stored_tpr_velo = []
    stored_fpr_naive = []
    stored_tpr_naive = []
    stored_fpr_LIN = []
    stored_tpr_LIN = []
    stored_recall_velo = []
    stored_precision_velo = []
    stored_recall_naive = []
    stored_precision_naive = []
    stored_recall_LIN = []
    stored_precision_LIN = []
    # Split into train and test: 
    # - Split into five folds 
    # - Stratified by relapse_label
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold_index, (train_index, test_index) in enumerate(skf.split(X, relapse_label)):
        print(f"Fold {fold_index}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        # Reset the index of X_train and X_test
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        patient_dictionary = get_ii_indexed_subset_dict(X_train, patient_dictionary_complete)
        patient_dictionary_test = get_ii_indexed_subset_dict(X_test, patient_dictionary_complete)
        X_train = X_train.copy().drop(columns=["USUBJID"])
        X_test = X_test.copy().drop(columns=["USUBJID"])

        # Create X_full and patient_dictionary_full by combining patient_dictionary and patient_dictionary_test (not clipped)
        N_patients_train, P = X_train.shape
        assert len(patient_dictionary) == N_patients_train, "len(patient_dictionary)"+str(len(patient_dictionary))
        N_patients_test, P = X_test.shape
        assert len(patient_dictionary_test) == N_patients_test, "len(patient_dictionary_test)"+str(len(patient_dictionary_test))
        assert X_train.shape[1] == X_test.shape[1] # P
        X_full = pd.concat([X_train, X_test])
        patient_dictionary_full = deepcopy(patient_dictionary)
        for ii in range(len(patient_dictionary_test)):
            patient_dictionary_full[ii+N_patients_train] = deepcopy(patient_dictionary_test[ii])

        # Store the true and predicted relapse times for each fold
        fold_cumul_binary_progress_or_not = np.array([])
        fold_cumul_new_p_progression_velo = np.array([])
        fold_cumul_new_p_progression = np.array([])
        fold_cumul_new_p_progression_LIN = np.array([])

        ##############
        # Scaling to keep numerics ok
        SCALE_TIME_AND_Y = False # Scale back for plotting! 
        time_scale = 1
        M_scale = 1
        if SCALE_TIME_AND_Y:
            df = pd.DataFrame(columns=["patient_id", "mprotein_value", "time"])
            for ii in range(len(patient_dictionary_full)):
                patient = patient_dictionary_full[ii]
                mprot = patient.Mprotein_values
                times = patient.measurement_times
                for jj in range(len(mprot)):
                    single_entry = pd.DataFrame({"patient_id":[ii], "mprotein_value":[mprot[jj]], "time":[times[jj]]})
                    df = pd.concat([df, single_entry], ignore_index=True)
            group_id = df["patient_id"].tolist()
            assert not np.isnan(group_id).any()
            assert not np.isinf(group_id).any()
            Y_flat_no_nans = np.array(df["mprotein_value"].tolist())
            assert min(Y_flat_no_nans) >= 0
            assert not np.isnan(Y_flat_no_nans).any()
            assert not np.isinf(Y_flat_no_nans).any()
            t_flat_no_nans = np.array(df["time"].tolist())
            assert min(t_flat_no_nans) >= 0
            assert not np.isnan(t_flat_no_nans).any()
            assert not np.isinf(t_flat_no_nans).any()
            N_patients = len(patient_dictionary_full)
            yi0 = np.zeros(N_patients)
            for ii in range(N_patients):
                yi0[ii] = max(patient_dictionary_full[ii].Mprotein_values[0], 1e-5)
            assert min(yi0) > 0 #Strictly greater than zero required because we log transform it for the log prior of psi 
            assert not np.isnan(yi0).any()
            assert not np.isinf(yi0).any()

            # Time transform! 
            t_max = np.amax(t_flat_no_nans)
            # Y transform! 
            Y_max = np.amax(Y_flat_no_nans)

            time_scale = t_max
            M_scale = Y_max

            for ii, patient in patient_dictionary_full.items():
                patient.Mprotein_values = patient.Mprotein_values / M_scale
                patient.measurement_times = patient.measurement_times / time_scale
                for covariate_name, time_series in patient.longitudinal_data.items():
                    patient.longitudinal_times[covariate_name] = patient.longitudinal_times[covariate_name] / time_scale
                patient.mrd_times = patient.mrd_times / time_scale
            for ii, patient in patient_dictionary.items():
                patient.Mprotein_values = patient.Mprotein_values / M_scale
                patient.measurement_times = patient.measurement_times / time_scale
                for covariate_name, time_series in patient.longitudinal_data.items():
                    patient.longitudinal_times[covariate_name] = patient.longitudinal_times[covariate_name] / time_scale
                patient.mrd_times = patient.mrd_times / time_scale
            for ii, patient in patient_dictionary_test.items():
                patient.Mprotein_values = patient.Mprotein_values / M_scale
                patient.measurement_times = patient.measurement_times / time_scale
                for covariate_name, time_series in patient.longitudinal_data.items():
                    patient.longitudinal_times[covariate_name] = patient.longitudinal_times[covariate_name] / time_scale
                patient.mrd_times = patient.mrd_times / time_scale
        # All patient measurement times and values are scaled from now on
        ##############

        # Clip test patients to create dictionary for fitting with partial M protein for test patients only
        patient_dictionary_fit = deepcopy(patient_dictionary)
        for ii in range(len(patient_dictionary_test)):
            clip_patient = deepcopy(patient_dictionary_test[ii])
            clip_patient.Mprotein_values = clip_patient.Mprotein_values[clip_patient.measurement_times <= CLIP_MPROTEIN_TIME]
            clip_patient.measurement_times = clip_patient.measurement_times[clip_patient.measurement_times <= CLIP_MPROTEIN_TIME]
            patient_dictionary_fit[ii+N_patients_train] = clip_patient
        assert len(patient_dictionary_full) == len(patient_dictionary_fit)
        assert X_full.shape[0] == len(patient_dictionary_fit)
        name_lin = "simdata_partial_Mprot_lin_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_CLIP_"+str(CLIP_MPROTEIN_TIME)+"_win_"+str(pred_window_length)+"_fold_"+str(fold_index)
        name_ind = "simdata_partial_Mprot_ind_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_CLIP_"+str(CLIP_MPROTEIN_TIME)+"_win_"+str(pred_window_length)+"_fold_"+str(fold_index)
        # Visualize parameter dependancy on covariates 
        #plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r)
        #plot_parameter_dependency_on_covariates(SAVEDIR, name_lin, X, expected_theta_1, true_theta_rho_s, true_rho_s)
        ind_model = individual_model(patient_dictionary_fit, name_ind, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, model_rho_r_dependancy_on_rho_s=model_rho_r_dependancy_on_rho_s)
        lin_model = linear_model(X_full, patient_dictionary_fit, name_lin, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, model_rho_r_dependancy_on_rho_s=model_rho_r_dependancy_on_rho_s)
        for model, name in [(ind_model, name_ind), (lin_model, name_lin)]:
            try:
                picklefile = open(SAVEDIR+name+"_idata_pickle", "rb")
                idata = pickle.load(picklefile)
                picklefile.close()
                print("Loading idata for " + name)
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
            if name == name_ind:
                quasi_geweke_test(idata, model_name="none", first=0.1, last=0.5)
                plot_posterior_traces(idata, SAVEDIR, name_ind, psi_prior, model_name="none")
            else:
                quasi_geweke_test(idata, model_name="linear", first=0.1, last=0.5)
                plot_posterior_traces(idata, SAVEDIR, name_ind, psi_prior, model_name="linear")
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
            if PLOTTING:
                plot_fit_and_predictions(idata, patient_dictionary_full, N_patients_train, SAVEDIR, name, y_resolution, CLIP_MPROTEIN_TIME, CI_with_obs_noise=False, PLOT_RESISTANT=False)
            del idata
            gc.collect()

        # Velocity model
        p_progression_velo = predict_PFS_velocity_model(patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon)
        print("p_progression_velo", p_progression_velo)

        # 5 Calculate predicted chance of PFS and plot AUC and AUPR
        true_pfs = get_true_pfs_new(patient_dictionary_test, time_scale=time_scale, M_scale=M_scale)
        print("True PFS\n", true_pfs)
        print("Average PFS among actual PFS", np.mean(true_pfs[true_pfs>0]))
        print("Std PFS among actual PFS", np.std(true_pfs[true_pfs>0]))
        print("Median PFS among actual PFS", np.median(true_pfs[true_pfs>0]))
        print("Max PFS among actual PFS", np.max(true_pfs[true_pfs>0]))
        print("Min PFS among actual PFS", np.min(true_pfs[true_pfs>0]))
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
            subset_true_pfs = []
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
                    subset_true_pfs.append(true_pfs[ii])
                    new_p_progression_velo.append(p_progression_velo[ii])
                    new_p_progression.append(p_progression[ii])
                    new_p_progression_LIN.append(p_progression_LIN[ii])
            print("N_patients_test originally", N_patients_test)
            print("len(patient_dictionary_test_progression)", len(patient_dictionary_test_progression))
            binary_progress_or_not = [1 if x > CLIP_MPROTEIN_TIME and x <= end_of_prediction_horizon else 0 for x in subset_true_pfs]
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
        stored_fpr_velo.append(fpr_velo)
        stored_tpr_velo.append(tpr_velo)
        stored_fpr_naive.append(fpr_naive)
        stored_tpr_naive.append(tpr_naive)
        stored_fpr_LIN.append(fpr_LIN)
        stored_tpr_LIN.append(tpr_LIN)
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

        # Cumulative AUC (Aggregated in fold):
        fold_cumul_binary_progress_or_not = np.append(fold_cumul_binary_progress_or_not, binary_progress_or_not)
        fold_cumul_new_p_progression_velo = np.append(fold_cumul_new_p_progression_velo, new_p_progression_velo)
        fold_cumul_new_p_progression = np.append(fold_cumul_new_p_progression, new_p_progression)
        fold_cumul_new_p_progression_LIN = np.append(fold_cumul_new_p_progression_LIN, new_p_progression_LIN)

        fold_cumul_fpr_velo, fold_cumul_tpr_velo, fold_cumul_threshold_velo = metrics.roc_curve(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression_velo) #(y_test, preds)
        fold_cumul_fpr_naive, fold_cumul_tpr_naive, fold_cumul_threshold_naive = metrics.roc_curve(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression) #(y_test, preds)
        fold_cumul_fpr_LIN, fold_cumul_tpr_LIN, fold_cumul_threshold_LIN = metrics.roc_curve(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression_LIN) #(y_test, preds)
        fold_cumul_roc_auc_velo = metrics.auc(fold_cumul_fpr_velo, fold_cumul_tpr_velo)
        fold_cumul_roc_auc_naive = metrics.auc(fold_cumul_fpr_naive, fold_cumul_tpr_naive)
        fold_cumul_roc_auc_LIN = metrics.auc(fold_cumul_fpr_LIN, fold_cumul_tpr_LIN)
        plt.figure()
        plt.grid(visible=True)
        plt.title("Cumulative ROC predicting 6 cycles ahead after "+str(pred_window_starts[0])+" to "+str(CLIP_MPROTEIN_TIME)+" days")
        plt.plot([0,1], [0,1], color='grey', linestyle='--', label='_nolegend_')
        for fi in range(fold_index+1):
            plt.plot(stored_fpr_velo[fi], stored_tpr_velo[fi], color=plt.cm.viridis(0.9), alpha=0.3, linestyle="--")
            plt.plot(stored_fpr_naive[fi], stored_tpr_naive[fi], color=plt.cm.viridis(0.6), alpha=0.3, linestyle="--")
            plt.plot(stored_fpr_LIN[fi], stored_tpr_LIN[fi], color=plt.cm.viridis(0.3), alpha=0.3, linestyle="--")
        plt.plot(fold_cumul_fpr_velo, fold_cumul_tpr_velo, color=plt.cm.viridis(0.9), label = 'Velocity model (AUC = %0.2f)' % np.mean(fold_cumul_roc_auc_velo), linewidth=3)
        plt.plot(fold_cumul_fpr_naive, fold_cumul_tpr_naive, color=plt.cm.viridis(0.6), label = 'No covariates (AUC = %0.2f)' % np.mean(fold_cumul_roc_auc_naive), linewidth=3)
        plt.plot(fold_cumul_fpr_LIN, fold_cumul_tpr_LIN, color=plt.cm.viridis(0.3), label = 'With covariates (AUC = %0.2f)' % np.mean(fold_cumul_roc_auc_LIN), linewidth=3)
        plt.legend(loc = 'lower right')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(SAVEDIR+name+"_Cumulative_AUC_from_"+str(pred_window_starts[0])+"__"+str(len(fold_cumul_binary_progress_or_not))+"_test_patients_"+str(sum(fold_cumul_binary_progress_or_not))+"_progressors.pdf", dpi=300)
        #plt.show()
        plt.close()
        a_file = open(SAVEDIR+name+"_fold_cumul_true", "wb")
        pickle.dump(fold_cumul_binary_progress_or_not, a_file)
        a_file.close()
        np.savetxt(SAVEDIR+name+"_fold_cumul_binary_progress_or_not"+".csv", fold_cumul_binary_progress_or_not, fmt="%s")
        a_file = open(SAVEDIR+name+"_fold_cumul_p_velo", "wb")
        pickle.dump(fold_cumul_new_p_progression_velo, a_file)
        a_file.close()
        np.savetxt(SAVEDIR+name+"_fold_cumul_new_p_progression_velo"+".csv", fold_cumul_new_p_progression_velo, fmt="%s")
        a_file = open(SAVEDIR+name+"_fold_cumul_p", "wb")
        pickle.dump(fold_cumul_new_p_progression, a_file)
        a_file.close()
        np.savetxt(SAVEDIR+name+"_fold_cumul_new_p_progression"+".csv", fold_cumul_new_p_progression, fmt="%s")
        a_file = open(SAVEDIR+name+"_fold_cumul_p_LIN", "wb")
        pickle.dump(fold_cumul_new_p_progression_LIN, a_file)
        a_file.close()
        np.savetxt(SAVEDIR+name+"_fold_cumul_new_p_progression_LIN"+".csv", fold_cumul_new_p_progression_LIN, fmt="%s")

        ## AUPR 
        proportion_progressions = sum(binary_progress_or_not) / len(binary_progress_or_not)
        precision_velo, recall_velo, threshold_velo = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression_velo) #(y_test, preds)
        precision_naive, recall_naive, threshold_naive = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression) #(y_test, preds)
        precision_LIN, recall_LIN, threshold_LIN = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression_LIN) #(y_test, preds)
        stored_recall_velo.append(recall_velo)
        stored_precision_velo.append(precision_velo)
        stored_recall_naive.append(recall_naive)
        stored_precision_naive.append(precision_naive)
        stored_recall_LIN.append(recall_LIN)
        stored_precision_LIN.append(precision_LIN)
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

        # Cumulative:
        fold_cumul_proportion_progressions = sum(fold_cumul_binary_progress_or_not) / len(fold_cumul_binary_progress_or_not)
        fold_cumul_precision_velo, fold_cumul_recall_velo, fold_cumul_threshold_velo = metrics.precision_recall_curve(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression_velo) #(y_test, preds)
        fold_cumul_precision_naive, fold_cumul_recall_naive, fold_cumul_threshold_naive = metrics.precision_recall_curve(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression) #(y_test, preds)
        fold_cumul_precision_LIN, fold_cumul_recall_LIN, fold_cumul_threshold_LIN = metrics.precision_recall_curve(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression_LIN) #(y_test, preds)
        fold_cumul_aupr_velo = metrics.average_precision_score(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression_velo)
        fold_cumul_aupr_naive = metrics.average_precision_score(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression)
        fold_cumul_aupr_LIN = metrics.average_precision_score(fold_cumul_binary_progress_or_not, fold_cumul_new_p_progression_LIN)
        plt.figure()
        plt.grid(visible=True)
        plt.title("Cumulative PR predicting 6 cycles ahead after "+str(pred_window_starts[0])+" to "+str(CLIP_MPROTEIN_TIME)+" days")
        plt.plot([0,1], [fold_cumul_proportion_progressions, fold_cumul_proportion_progressions], color='grey', linestyle='--', label='_nolegend_')
        for fi in range(fold_index+1):
            plt.plot(stored_recall_velo[fi], stored_precision_velo[fi], color=plt.cm.viridis(0.9), alpha=0.3, linestyle="--")
            plt.plot(stored_recall_naive[fi], stored_precision_naive[fi], color=plt.cm.viridis(0.6), alpha=0.3, linestyle="--")
            plt.plot(stored_recall_LIN[fi], stored_precision_LIN[fi], color=plt.cm.viridis(0.3), alpha=0.3, linestyle="--")
        plt.plot(fold_cumul_recall_velo, fold_cumul_precision_velo, color=plt.cm.viridis(0.9), label = 'Velocity model (AUPR = %0.2f)' % np.mean(fold_cumul_aupr_velo), linewidth=3)
        plt.plot(fold_cumul_recall_naive, fold_cumul_precision_naive, color=plt.cm.viridis(0.6), label = 'No covariates (AUPR = %0.2f)' % np.mean(fold_cumul_aupr_naive), linewidth=3)
        plt.plot(fold_cumul_recall_LIN, fold_cumul_precision_LIN, color=plt.cm.viridis(0.3), label = 'With covariates (AUPR = %0.2f)' % np.mean(fold_cumul_aupr_LIN), linewidth=3)
        plt.legend(loc = 'lower right')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('Precision')
        plt.xlabel('Recall (True positive rate)')
        plt.savefig(SAVEDIR+name+"_Cumulative_AUPR_from_"+str(pred_window_starts[0])+"__"+str(len(fold_cumul_binary_progress_or_not))+"_test_patients_"+str(sum(fold_cumul_binary_progress_or_not))+"_progressors.pdf", dpi=300)
        #plt.show()
        plt.close()

        del binary_progress_or_not
        del new_p_progression
        del new_p_progression_LIN
        del new_p_progression_velo
        del patient_dictionary_fit
        gc.collect()

        """

        ## Generate test patients
        #N_patients_test = 50
        #test_seed = 23
        #X_test, patient_dictionary_test, parameter_dictionary_test, expected_theta_1_test, true_theta_rho_s_test, true_rho_s_test = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_test, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=test_seed, RANDOM_EFFECTS=RANDOM_EFFECTS_TEST)
        #print("Done generating test patients")

        #plot_all_credible_intervals(name_ind, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name_ind, y_resolution, model_name=model_name, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=True, PARALLELLIZE=True)
        """
