from utilities import *
import arviz as az
import pymc as pm
# One BNN with 3 outputs 
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
##############################
# Function argument shapes: 
# X is an (N_patients, P) shaped pandas dataframe
# patient dictionary contains N_patients patients in the same order as X
def joint_BNN_model(X, patient_dictionary, name, psi_prior="lognormal", MODEL_RANDOM_EFFECTS=True, FUNNEL_REPARAMETRIZATION=False, FUNNEL_WEIGHTS = False, WEIGHT_PRIOR = "symmetry_fix", SAVING=False, n_hidden = 3, net_list=["pi", "rho_r", "rho_s"]):
    df = pd.DataFrame(columns=["patient_id", "mprotein_value", "time"])
    for ii in range(len(patient_dictionary)):
        patient = patient_dictionary[ii]
        mprot = patient.Mprotein_values
        times = patient.measurement_times
        for jj in range(len(mprot)):
            single_entry = pd.DataFrame({"patient_id":[ii], "mprotein_value":[mprot[jj]], "time":[times[jj]]})
            df = pd.concat([df, single_entry], ignore_index=True)
    group_id = df["patient_id"].tolist()
    Y_flat_no_nans = np.array(df["mprotein_value"].tolist())
    t_flat_no_nans = np.array(df["time"].tolist())

    N_patients, P = X.shape
    P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    X_not_transformed = X.copy()
    X = X.T
    #Y = np.transpose(np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()]))
    #t = np.transpose(np.array([patient.measurement_times for _, patient in patient_dictionary.items()]))
    #Y = np.empty((N_patients, max([len(patient.Mprotein_values) for _, patient in patient_dictionary.items()])))
    #Y[:] = np.nan 
    #for ii, mprot in enumerate([patient.Mprotein_values for _, patient in patient_dictionary.items()]):
    #    Y[ii,0:len(mprot)] = mprot
    #Y = np.transpose(Y)

    #t = np.empty((N_patients, max([len(patient.measurement_times) for _, patient in patient_dictionary.items()])))
    #t[:] = np.nan 
    #for ii, mtimes in enumerate([patient.measurement_times for _, patient in patient_dictionary.items()]):
    #    t[ii,0:len(mtimes)] = mtimes
    #t = np.transpose(t)
    #assert t.shape == Y.shape

    yi0 = np.zeros(N_patients)
    for ii in range(N_patients):
        yi0[ii] = patient_dictionary[ii].Mprotein_values[0]
    assert yi0.min() > 0, "Initial M protein values yi0 must be positive due to lognormal prior on psi"
    # Dimensions: 
    # X: (P, N_cases)
    # y: (M_max, N)
    # t: (M_max, N)
    """
    viz_Y = Y[Y<250]
    plt.figure()
    sns.distplot(Y, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1, 'gridsize':100})
    if SAVING:
        plt.savefig("./plots/posterior_plots/"+name+"-plot_density.png")
    plt.close
    plt.figure()
    sns.distplot(viz_Y, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1, 'gridsize':100})
    if SAVING:
        plt.savefig("./plots/posterior_plots/"+name+"-plot_density_lessthan_250.png")
    plt.close
    """
    if psi_prior not in ["lognormal", "normal"]:
        print("Unknown prior option specified for psi; Using 'lognormal' prior")
        psi_prior = "lognormal"

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[0], n_hidden)
    if WEIGHT_PRIOR == "iso_normal":
        init_out = np.random.randn(n_hidden,3)
    else:
        #init_out = np.random.exponential(lam=10, size=n_hidden) # scale=0.1
        init_out = abs(np.random.randn(n_hidden,3))

    with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as neural_net_model:
        # Observation noise (std)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
        #log_sigma_obs = pm.Normal("log_sigma_obs", mu=2*np.log(0.01), sigma=2**2)
        #sigma_obs = pm.Deterministic("sigma_obs", np.exp(log_sigma_obs))

        # alpha
        alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)
        ## alpha gone. Did not work either. Still nan. 
        ###alpha_offset = pm.Exponential("alpha_offset", lam=10, size=3) # dims=3?

        #sigma_weights_in = pm.HalfNormal("sigma_weights_in", sigma=0.1)
        #sigma_weights_in = pm.HalfNormal("sigma_weights_in", sigma=0.1, shape=(X.shape[0], 1))
        log_sigma_weights_in = pm.Normal("log_sigma_weights_in", mu=2*np.log(0.01), sigma=2.5**2, shape=(X.shape[0], 1))
        sigma_weights_in = pm.Deterministic("sigma_weights_in", np.exp(log_sigma_weights_in))
        if FUNNEL_WEIGHTS == True: # Funnel reparametrized weights: 
            # Weights input to 1st layer
            weights_in_offset = pm.Normal("weights_in_offset ", mu=0, sigma=1, shape=(X.shape[0], n_hidden))
            #weights_in = pm.Deterministic(("weights_in", weights_in_offset * sigma_weights_in))
            weights_in = pm.Deterministic("weights_in", weights_in_offset * np.repeat(sigma_weights_in, n_hidden, axis=1))
            # Weights from 1st to 2nd layer
            if WEIGHT_PRIOR == "iso_normal":
                weights_out_offset = pm.Normal("weights_out_offset ", mu=0, sigma=1, shape=(n_hidden,3))
            # WEIGHT_PRIOR == "Student_out" does not make sense with funnel
            else: # Handling symmetry
                weights_out_offset = pm.HalfNormal("weights_out_offset ", sigma=1, shape=(n_hidden,3))
            sigma_weights_out = pm.HalfNormal("sigma_weights_out", sigma=0.1)
            weights_out = pm.Deterministic("weights_out", weights_out_offset * sigma_weights_out)
        else:
            # Weights input to 1st layer
            if WEIGHT_PRIOR == "Horseshoe":
                # Global shrinkage prior
                tau_in = pm.HalfStudentT("tau_in", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                # Local shrinkage prior
                lam_in = pm.HalfStudentT("lam_in", 2, shape=(X.shape[0], n_hidden)) #dims
                c2_in = pm.InverseGamma("c2_in", 1, 0.1)
                z_in = pm.Normal("z_in", 0.0, 1.0, shape=(X.shape[0], n_hidden)) #dims
                # Shrunken coefficients
                weights_in = pm.Deterministic("weights_in", z_in * tau_in * lam_in * np.sqrt(c2_in / (c2_in + tau_in**2 * lam_in**2))) # dims
            else: 
                weights_in = pm.Normal('weights_in', 0, sigma=np.repeat(sigma_weights_in, n_hidden, axis=1), shape=(X.shape[0], n_hidden), initval=init_1)
            # Weights from 1st to 2nd layer
            if WEIGHT_PRIOR == "iso_normal":
                sigma_weights_out = pm.HalfNormal("sigma_weights_out", sigma=0.1)
                weights_out = pm.Normal('weights_out', 0, sigma=sigma_weights_out, shape=(n_hidden,3), initval=init_out)
            elif WEIGHT_PRIOR == "Student_out": # Handling symmetry
                weights_out = pm.HalfStudentT('weights_out', nu=4, sigma=1, shape=(n_hidden,3), initval=init_out)
            elif WEIGHT_PRIOR == "Horseshoe":
                # Global shrinkage prior
                tau_out = pm.HalfStudentT("tau_out", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                # Local shrinkage prior
                lam_out = pm.HalfStudentT("lam_out", 2, shape=(n_hidden,3)) #dims
                c2_out = pm.InverseGamma("c2_out", 1, 0.1)
                z_out = pm.Normal("z_out", 0.0, 1.0, shape=(n_hidden,3)) #dims
                # Shrunken coefficients
                weights_out = pm.Deterministic("weights_out", z_out * tau_out * lam_out * np.sqrt(c2_out / (c2_out + tau_out**2 * lam_out**2))) # dims
            else: # Handling symmetry
                sigma_weights_out = pm.HalfNormal("sigma_weights_out", sigma=0.1)
                weights_out = pm.HalfNormal('weights_out', sigma=sigma_weights_out, shape=(n_hidden,3), initval=init_out)

        # offsets for each node between each layer 
        sigma_bias_in = pm.HalfNormal("sigma_bias_in", sigma=1, shape=(1,n_hidden))
        bias_in = pm.Normal("bias_in", mu=0, sigma=sigma_bias_in, shape=(1,n_hidden)) # sigma=sigma_bias_in
        
        # Calculate Y using neural net 
        # Leaky RELU activation
        pre_act_1 = pm.math.dot(X_not_transformed, weights_in) + bias_in
        act_1 = pm.math.switch(pre_act_1 > 0, pre_act_1, pre_act_1 * 0.01)

        # Output activation function is just unit transform for prediction model
        act_out = pm.math.dot(act_1, weights_out)

        # Latent variables theta
        omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
        if MODEL_RANDOM_EFFECTS:
            if FUNNEL_REPARAMETRIZATION == True: 
                # Reparametrized to escape/explore the funnel of Hell (https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/):
                theta_rho_s_offset = pm.Normal('theta_rho_s_offset', mu=0, sigma=1, shape=N_patients)
                theta_rho_r_offset = pm.Normal('theta_rho_r_offset', mu=0, sigma=1, shape=N_patients)
                theta_pi_r_offset  = pm.Normal('theta_pi_r_offset',  mu=0, sigma=1, shape=N_patients)
                theta_rho_s = pm.Deterministic("theta_rho_s", (alpha[0] + act_out[:,0] + theta_rho_s_offset * omega[0]))
                theta_rho_r = pm.Deterministic("theta_rho_r", (alpha[1] + act_out[:,1] + theta_rho_r_offset * omega[1]))
                theta_pi_r  = pm.Deterministic("theta_pi_r",  (alpha[2] + act_out[:,2] + theta_pi_r_offset  * omega[2]))
            else:
                # Original
                theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + act_out[:,0], sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
                theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + act_out[:,1], sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
                theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + act_out[:,2],  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X
        else: 
            theta_rho_s = pm.Deterministic("theta_rho_s", alpha[0] + act_out[:,0])
            theta_rho_r = pm.Deterministic("theta_rho_r", alpha[1] + act_out[:,1])
            theta_pi_r  = pm.Deterministic("theta_pi_r",  alpha[2] + act_out[:,2])

        # psi: True M protein at time 0
        # 1) Normal. Fast convergence, but possibly negative tail 
        if psi_prior=="normal":
            psi = pm.Normal("psi", mu=yi0, sigma=sigma_obs, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        # 2) Lognormal. Works if you give it time to converge
        if psi_prior=="lognormal":
            xi = pm.HalfNormal("xi", sigma=1)
            log_psi = pm.Normal("log_psi", mu=np.log(yi0+1e-8), sigma=xi, shape=N_patients)
            psi = pm.Deterministic("psi", np.exp(log_psi))

        # Transformed latent variables 
        rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
        rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
        pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

        # Observation model 
        #mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

        # Likelihood (sampling distribution) of observations
        # Check for nan in Y and t; and only use 
        #mu_Y = mu_Y[~np.isnan(t)]
        mu_Y = psi[group_id] * (pi_r[group_id]*pm.math.exp(rho_r[group_id]*t_flat_no_nans) + (1-pi_r[group_id])*pm.math.exp(rho_s[group_id]*t_flat_no_nans))
        #Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y[~np.isnan(t)])
        #Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y)
    # Visualize model
    #import graphviz 
    #gv = pm.model_to_graphviz(neural_net_model) # With shared vcariables: --> 170 assert force_compile or (version == get_version())   AssertionError.
    #gv.render(filename="./plots/posterior_plots/"+name+"_graph_of_model", format="png", view=False)
    # Sample from prior:
    """
    with neural_net_model:
        prior_samples = pm.sample_prior_predictive(200)
    raveled_Y_true = np.ravel(Y)
    raveled_Y_sample = np.ravel(prior_samples.prior_predictive["Y_obs"])
    # Below plotlimit_prior
    plotlimit_prior = 1000
    plt.figure()
    az.plot_dist(raveled_Y_true[raveled_Y_true<plotlimit_prior], color="C1", label="observed", bw=3)
    az.plot_dist(raveled_Y_sample[raveled_Y_sample<plotlimit_prior], label="simulated", bw=3)
    plt.title("Samples from prior compared to observations, for Y<plotlimit_prior")
    plt.xlabel("Y (M protein)")
    plt.ylabel("Frequency")
    if SAVING:
        plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples_below_"+str(plotlimit_prior)+".png")
    plt.close()
    # All samples: 
    plt.figure()
    az.plot_dist(raveled_Y_true, color="C1", label="observed", bw=3)
    az.plot_dist(raveled_Y_sample, label="simulated", bw=3)
    plt.title("Samples from prior compared to observations")
    plt.xlabel("Y (M protein)")
    plt.ylabel("Frequency")
    if SAVING:
        plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples.png")
    plt.close()
    """
    return neural_net_model
