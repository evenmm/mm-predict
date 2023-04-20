from utilities import *
import arviz as az
import pymc as pm
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
##############################
# Function argument shapes: 
# X is an (N_patients, P) shaped pandas dataframe
# patient dictionary contains N_patients patients in the same order as X

def linear_model(X, patient_dictionary, name, N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, psi_prior="lognormal", FUNNEL_REPARAMETRIZATION=False, method="HMC"):
    df = pd.DataFrame(columns=["patient_id", "mprotein_value", "time"])
    for ii in range(len(patient_dictionary)):
        patient = patient_dictionary[ii]
        mprot = patient.Mprotein_values
        times = patient.measurement_times
        for jj in range(len(mprot)):
            single_entry = pd.DataFrame({"patient_id":[ii], "mprotein_value":[mprot[jj]], "time":[times[jj]]})
            df = pd.concat([df, single_entry], ignore_index=True)
    print(df.head(n=6))

    # Experimental:
    group_id = df["patient_id"].tolist()
    Y_flat_no_nans = df["mprotein_value"].tolist()
    t_flat_no_nans = df["time"].tolist()

    ## patient_id_list
    #df['group'] = pd.Categorical(df['patient_id'], ordered = False)
    #group_id = df['group'].cat.codes.values # This one could contain more than just 0 to N, it can be separate for train and test 
    #print(group_id)

    N_patients, P = X.shape
    P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    X_not_transformed = X.copy()
    X = X.T
    """
    #Y = np.transpose(np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()]))
    #t = np.transpose(np.array([patient.measurement_times for _, patient in patient_dictionary.items()]))
    Y = np.empty((N_patients, max([len(patient.Mprotein_values) for _, patient in patient_dictionary.items()])))
    Y[:] = np.nan 
    #for ii, mprot in enumerate([patient.Mprotein_values for _, patient in patient_dictionary.items()]):
    for ii in range(N_patients):
        mprot = patient_dictionary[ii].Mprotein_values
        Y[ii,0:len(mprot)] = mprot
    Y = np.transpose(Y)
    t = np.empty((N_patients, max([len(patient.measurement_times) for _, patient in patient_dictionary.items()])))
    t[:] = np.nan 
    #for ii, mtimes in enumerate([patient.measurement_times for _, patient in patient_dictionary.items()]):
    for ii in range(N_patients):
        mtimes = patient_dictionary[ii].measurement_times
        t[ii,0:len(mtimes)] = mtimes
    t = np.transpose(t)
    #assert t.shape == Y.shape
    """
    #yi0 = np.maximum(1e-5, np.array([patient.Mprotein_values[0] for _, patient in patient_dictionary.items()]))
    yi0 = np.zeros(N_patients)
    for ii in range(N_patients):
        yi0[ii] = patient_dictionary[ii].Mprotein_values[0]

    #print("Max(Y):", np.amax(Y))
    #print("Max(t):", np.amax(t))
    #viz_Y = Y[Y<250]
    #plt.figure()
    #sns.distplot(Y, hist=True, kde=True, 
    #         bins=int(180/5), color = 'darkblue', 
    #         hist_kws={'edgecolor':'black'},
    #         kde_kws={'linewidth': 1, 'gridsize':100})
    #plt.savefig("./plots/posterior_plots/"+name+"-plot_density.png")
    #plt.close
    #plt.figure()
    #sns.distplot(viz_Y, hist=True, kde=True, 
    #         bins=int(180/5), color = 'darkblue', 
    #         hist_kws={'edgecolor':'black'},
    #         kde_kws={'linewidth': 1, 'gridsize':100})
    #plt.savefig("./plots/posterior_plots/"+name+"-plot_density_lessthan_250.png")
    #plt.close
    if psi_prior not in ["lognormal", "normal"]:
        print("Unknown prior option specified for psi; Using 'lognormal' prior")
        psi_prior = "lognormal"

    with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as linear_model:
        # Observation noise (std)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

        # alpha
        alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)

        # beta (with horseshoe priors):
        # Global shrinkage prior
        tau_rho_s = pm.HalfStudentT("tau_rho_s", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
        tau_rho_r = pm.HalfStudentT("tau_rho_r", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
        tau_pi_r = pm.HalfStudentT("tau_pi_r", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
        # Local shrinkage prior
        lam_rho_s = pm.HalfStudentT("lam_rho_s", 2, dims="predictors") # shape = (P,) without predictors
        lam_rho_r = pm.HalfStudentT("lam_rho_r", 2, dims="predictors")
        lam_pi_r = pm.HalfStudentT("lam_pi_r", 2, dims="predictors")
        c2_rho_s = pm.InverseGamma("c2_rho_s", 1, 0.1)
        c2_rho_r = pm.InverseGamma("c2_rho_r", 1, 0.1)
        c2_pi_r = pm.InverseGamma("c2_pi_r", 1, 0.1)
        z_rho_s = pm.Normal("z_rho_s", 0.0, 1.0, dims="predictors")
        z_rho_r = pm.Normal("z_rho_r", 0.0, 1.0, dims="predictors")
        z_pi_r = pm.Normal("z_pi_r", 0.0, 1.0, dims="predictors")
        # Shrunken coefficients
        beta_rho_s = pm.Deterministic("beta_rho_s", z_rho_s * tau_rho_s * lam_rho_s * np.sqrt(c2_rho_s / (c2_rho_s + tau_rho_s**2 * lam_rho_s**2)), dims="predictors")
        beta_rho_r = pm.Deterministic("beta_rho_r", z_rho_r * tau_rho_r * lam_rho_r * np.sqrt(c2_rho_r / (c2_rho_r + tau_rho_r**2 * lam_rho_r**2)), dims="predictors")
        beta_pi_r = pm.Deterministic("beta_pi_r", z_pi_r * tau_pi_r * lam_pi_r * np.sqrt(c2_pi_r / (c2_pi_r + tau_pi_r**2 * lam_pi_r**2)), dims="predictors")

        # Latent variables theta
        omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
        if FUNNEL_REPARAMETRIZATION == True: 
            # Reparametrized to escape/explore the funnel of Hell (https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/):
            theta_rho_s_offset = pm.Normal('theta_rho_s_offset', mu=0, sigma=1, shape=N_patients)
            theta_rho_r_offset = pm.Normal('theta_rho_r_offset', mu=0, sigma=1, shape=N_patients)
            theta_pi_r_offset  = pm.Normal('theta_pi_r_offset',  mu=0, sigma=1, shape=N_patients)
            theta_rho_s = pm.Deterministic("theta_rho_s", (alpha[0] + pm.math.dot(beta_rho_s, X)) + theta_rho_s_offset * omega[0])
            theta_rho_r = pm.Deterministic("theta_rho_r", (alpha[1] + pm.math.dot(beta_rho_r, X)) + theta_rho_r_offset * omega[1])
            theta_pi_r  = pm.Deterministic("theta_pi_r",  (alpha[2] + pm.math.dot(beta_pi_r,  X)) + theta_pi_r_offset  * omega[2])
        else: 
            # Original
            theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + pm.math.dot(beta_rho_s, X), sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
            theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + pm.math.dot(beta_rho_r, X), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
            theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + pm.math.dot(beta_pi_r, X),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

        # psi: True M protein at time 0
        # 1) Normal. Fast convergence, but possibly negative tail 
        if psi_prior=="normal":
            psi = pm.Normal("psi", mu=yi0, sigma=sigma_obs, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        # 2) Lognormal. Works if you give it time to converge
        if psi_prior=="lognormal":
            xi = pm.HalfNormal("xi", sigma=1)
            log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=xi, shape=N_patients)
            psi = pm.Deterministic("psi", np.exp(log_psi))

        # Transformed latent variables 
        rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
        rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
        pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

        # Observation model 
        #t_no_nans = t[~np.isnan(t)]
        
        ##mu_Y = []
        ##for ii in range(N_patients):
        ##    times = patient_dictionary[ii].measurement_times
        ##    nonzero_Y = psi[ii] * (pi_r[ii]*np.exp(rho_r[ii]*times) + (1-pi_r[ii])*np.exp(rho_s[ii]*times))
        ##    mu_Y.append(nonzero_Y)
        ##mu_Y = np.array(mu_Y).flatten()
        #
        #mu_Y = [] # is a list of 
        #for ii in range(N_patients):
        #    times = patient_dictionary[ii].measurement_times
        #    for tt in times:
        #        nonzero_Y = psi[ii] * (pi_r[ii]*np.exp(rho_r[ii]*tt) + (1-pi_r)*np.exp(rho_s[ii]*tt))
        #        mu_Y.append(nonzero_Y)
        ##mu_Y = np.array(mu_Y)

        #t_flat_no_nans = t[~np.isnan(t)]
        mu_Y = psi[group_id] * (pi_r[group_id]*np.exp(rho_r[group_id]*t_flat_no_nans) + (1-pi_r[group_id])*np.exp(rho_s[group_id]*t_flat_no_nans))
        Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y_flat_no_nans)

        #mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))
        #mu_Y = mu_Y[~np.isnan(t)] #this must be done in pymc style 

        # Likelihood (sampling distribution) of observations
        #print(Y[~np.isnan(t)])
        #print(np.isnan(Y[~np.isnan(t)]).any())
        #Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y[~np.isnan(t)])
    return linear_model
