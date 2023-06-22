from utilities import *

def predict_PFS(args): # Predicts observations of M protein
    sample_shape, ii, idata, X_test, patient_dictionary_test, N_rand_eff_pred, N_rand_obs_pred, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, evaluation_time = args
    if not CI_with_obs_noise:
        N_rand_eff_pred = N_rand_eff_pred * N_rand_obs_pred
        N_rand_obs_pred = 1
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    np.random.seed(ii) # Seeding the randomness in observation noise sigma, in random effects and in psi = yi0 + random(sigma)

    patient = patient_dictionary_test[ii]
    measurement_times = patient.get_measurement_times() 
    Mprotein_values = patient.get_Mprotein_values()
    treatment_history = patient.get_treatment_history()
    first_time = min(measurement_times[0], treatment_history[0].start)
    max_time = find_max_time(measurement_times)
    #test_times = np.array([30, 60, 90, 120, 150, 180])
    test_times = np.array([evaluation_time])
    y_resolution = len(test_times)
    predicted_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
    all_predicted_y_values_noiseless = np.empty(shape=(n_chains*N_rand_eff_pred, n_samples, y_resolution))
    for ch in range(n_chains):
        for sa in range(n_samples):
            sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
            alpha = np.ravel(idata.posterior['alpha'][ch,sa])

            if model_name == "linear": 
                this_beta_rho_s = np.ravel(idata.posterior['beta_rho_s'][ch,sa])
                this_beta_rho_r = np.ravel(idata.posterior['beta_rho_r'][ch,sa])
                this_beta_pi_r = np.ravel(idata.posterior['beta_pi_r'][ch,sa])
            elif model_name == "BNN": 
                # weights 
                weights_in_rho_s = idata.posterior['weights_in_rho_s'][ch,sa]
                weights_in_rho_r = idata.posterior['weights_in_rho_r'][ch,sa]
                weights_in_pi_r = idata.posterior['weights_in_pi_r'][ch,sa]
                weights_out_rho_s = idata.posterior['weights_out_rho_s'][ch,sa]
                weights_out_rho_r = idata.posterior['weights_out_rho_r'][ch,sa]
                weights_out_pi_r = idata.posterior['weights_out_pi_r'][ch,sa]

                # intercepts
                #sigma_bias_in = idata.posterior['sigma_bias_in'][ch,sa]
                bias_in_rho_s = np.ravel(idata.posterior['bias_in_rho_s'][ch,sa])
                bias_in_rho_r = np.ravel(idata.posterior['bias_in_rho_r'][ch,sa])
                bias_in_pi_r = np.ravel(idata.posterior['bias_in_pi_r'][ch,sa])

                pre_act_1_rho_s = np.dot(X_test.iloc[ii,:], weights_in_rho_s) + bias_in_rho_s
                pre_act_1_rho_r = np.dot(X_test.iloc[ii,:], weights_in_rho_r) + bias_in_rho_r
                pre_act_1_pi_r  = np.dot(X_test.iloc[ii,:], weights_in_pi_r)  + bias_in_pi_r

                act_1_rho_s = np.select([pre_act_1_rho_s > 0, pre_act_1_rho_s <= 0], [pre_act_1_rho_s, pre_act_1_rho_s*0.01], 0)
                act_1_rho_r = np.select([pre_act_1_rho_r > 0, pre_act_1_rho_r <= 0], [pre_act_1_rho_r, pre_act_1_rho_r*0.01], 0)
                act_1_pi_r =  np.select([pre_act_1_pi_r  > 0, pre_act_1_pi_r  <= 0], [pre_act_1_pi_r,  pre_act_1_pi_r*0.01],  0)

                # Output
                act_out_rho_s = np.dot(act_1_rho_s, weights_out_rho_s)
                act_out_rho_r = np.dot(act_1_rho_r, weights_out_rho_r)
                act_out_pi_r =  np.dot(act_1_pi_r,  weights_out_pi_r)

            elif model_name == "joint_BNN": 
                # weights 
                weights_in = idata.posterior['weights_in'][ch,sa]
                weights_out = idata.posterior['weights_out'][ch,sa]

                # intercepts
                #sigma_bias_in = idata.posterior['sigma_bias_in'][ch,sa]
                bias_in = np.ravel(idata.posterior['bias_in'][ch,sa])

                pre_act_1 = np.dot(X_test.iloc[ii,:], weights_in) + bias_in

                act_1 = np.select([pre_act_1 > 0, pre_act_1 <= 0], [pre_act_1, pre_act_1*0.01], 0)

                # Output
                act_out = np.dot(act_1, weights_out)
                act_out_rho_s = act_out[0]
                act_out_rho_r = act_out[1]
                act_out_pi_r =  act_out[2]

            # Random effects 
            omega  = np.ravel(idata.posterior['omega'][ch,sa])
            for ee in range(N_rand_eff_pred):
                if model_name == "linear":
                    #if MODEL_RANDOM_EFFECTS: 
                    predicted_theta_1 = np.random.normal(alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s), omega[0])
                    predicted_theta_2 = np.random.normal(alpha[1] + np.dot(X_test.iloc[ii,:], this_beta_rho_r), omega[1])
                    predicted_theta_3 = np.random.normal(alpha[2] + np.dot(X_test.iloc[ii,:], this_beta_pi_r), omega[2])
                    #else: 
                    #    predicted_theta_1 = alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s)
                    #    predicted_theta_2 = alpha[1] + np.dot(X_test.iloc[ii,:], this_beta_rho_r)
                    #    predicted_theta_3 = alpha[2] + np.dot(X_test.iloc[ii,:], this_beta_pi_r)
                elif model_name == "BNN" or model_name == "joint_BNN":
                    if MODEL_RANDOM_EFFECTS:
                        predicted_theta_1 = np.random.normal(alpha[0] + act_out_rho_s, omega[0])
                        predicted_theta_2 = np.random.normal(alpha[1] + act_out_rho_r, omega[1])
                        predicted_theta_3 = np.random.normal(alpha[2] + act_out_pi_r, omega[2])
                    else: 
                        predicted_theta_1 = alpha[0] + act_out_rho_s
                        predicted_theta_2 = alpha[1] + act_out_rho_r
                        predicted_theta_3 = alpha[2] + act_out_pi_r

                predicted_rho_s = - np.exp(predicted_theta_1)
                predicted_rho_r = np.exp(predicted_theta_2)
                predicted_pi_r  = 1/(1+np.exp(-predicted_theta_3))

                this_psi = patient.Mprotein_values[0] + np.random.normal(0,sigma_obs)
                predicted_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=predicted_pi_r, g_r=predicted_rho_r, g_s=predicted_rho_s, k_1=0, sigma=sigma_obs)
                these_parameters = predicted_parameters[ch,sa]
                # Predicted total and resistant M protein
                predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, test_times, treatment_history)
                all_predicted_y_values_noiseless[N_rand_eff_pred*ch + ee, sa] = predicted_y_values_noiseless
    flat_pred_y_values_noiseless = np.reshape(all_predicted_y_values_noiseless, (n_chains*n_samples*N_rand_eff_pred, y_resolution))
    # Predicted probability of recurrence
    # at 6 months = 180 days
    predicted_at_x_months = flat_pred_y_values_noiseless[:,-1]
    # The proportion of samples at that day that is above the first observed M protein value 
    p_recurrence = len(predicted_at_x_months[predicted_at_x_months > Mprotein_values[0]]) / len(predicted_at_x_months)
    return p_recurrence #predicted_PFS, point_predicted_PFS

def pfs_auc(evaluation_time, patient_dictionary_test, N_patients_test, idata, X_test, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, SAVEDIR, name):
    import sklearn.metrics as metrics
    N_patients_test = len(patient_dictionary_test)

    recurrence_or_not = np.zeros(N_patients_test)
    for ii in range(N_patients_test):
        patient = patient_dictionary_test[ii]
        mprot = patient.Mprotein_values
        times = patient.measurement_times
        recurrence_or_not[ii] = int( (mprot[1:][times[1:] < evaluation_time] > mprot[0]).any() )
    print("Recurrence", recurrence_or_not)
    print("Proportion with recurrence:", np.mean(recurrence_or_not))

    sample_shape = idata.posterior['psi'].shape # [chain, n_samples, dim]
    N_samples = sample_shape[1]
    # Posterior predictive CI for test data
    if N_samples <= 10:
        N_rand_eff_pred = 100 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 100:
        N_rand_eff_pred = 10 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 1000:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    else:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 10 # Number of observation noise samples to draw for each parameter sample 

    args = [(sample_shape, ii, idata, X_test, patient_dictionary_test, N_rand_eff_pred, N_rand_obs_pred, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, evaluation_time) for ii in range(N_patients_test)]

    p_recurrence = np.zeros(N_patients_test) 
    for ii, elem in enumerate(args):
        p_recurrence[ii] = predict_PFS(elem)

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(recurrence_or_not, p_recurrence) #(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    print(fpr)
    print(tpr)
    print(threshold)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(SAVEDIR+"AUC_"+str(N_patients_test)+"_test_patients_"+name+".pdf")
    plt.show()

evaluation_time = 1000 # days
pfs_auc(evaluation_time, patient_dictionary_test, N_patients_test, idata, X_test, model_name, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, SAVEDIR, name)
