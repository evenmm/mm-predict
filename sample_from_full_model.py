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
##############################
# Function argument shapes: 
# X is an (N_patients, P) shaped pandas dataframe
# patient dictionary contains N_patients patients in the same order as X
def sample_from_full_model(X, patient_dictionary, name, N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, psi_prior="lognormal", FUNNEL_REPARAMETRIZATION=False, method="HMC"):
    N_patients, P = X.shape
    P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    X_not_transformed = X.copy()
    X = X.T
    Y = np.transpose(np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()]))
    t = np.transpose(np.array([patient.measurement_times for _, patient in patient_dictionary.items()]))
    yi0 = np.maximum(1e-5, np.array([patient.Mprotein_values[0] for _, patient in patient_dictionary.items()]))
    # Dimensions: 
    # X: (P, N:cases)
    # y: (M_max, N)
    # t: (M_max, N)

    print("Max(Y):", np.amax(Y))
    print("Max(t):", np.amax(t))
    viz_Y = Y[Y<250]
    plt.figure()
    sns.distplot(Y, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1, 'gridsize':100})
    plt.savefig("./plots/posterior_plots/"+name+"-plot_density.png")
    plt.close
    plt.figure()
    sns.distplot(viz_Y, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1, 'gridsize':100})
    plt.savefig("./plots/posterior_plots/"+name+"-plot_density_lessthan_250.png")
    plt.close
    if psi_prior not in ["lognormal", "normal"]:
        print("Unknown prior option specified for psi; Using 'lognormal' prior")
        psi_prior = "lognormal"

    with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as multiple_patients_model:
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
        if FUNNEL_REPARAMETRIZATION == True: 
            # Reparametrized to escape/explore the funnel of Hell (https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/):
            theta_rho_s_offset = pm.Normal('theta_rho_s_offset', mu=0, sigma=1, shape=N_patients)
            theta_rho_r_offset = pm.Normal('theta_rho_r_offset', mu=0, sigma=1, shape=N_patients)
            theta_pi_r_offset  = pm.Normal('theta_pi_r_offset',  mu=0, sigma=1, shape=N_patients)
            theta_rho_s = pm.Deterministic("theta_rho_s", (alpha[0] + at.dot(beta_rho_s, X)) + theta_rho_s_offset * omega[0])
            theta_rho_r = pm.Deterministic("theta_rho_r", (alpha[1] + at.dot(beta_rho_r, X)) + theta_rho_r_offset * omega[1])
            theta_pi_r  = pm.Deterministic("theta_pi_r",  (alpha[2] + at.dot(beta_pi_r,  X)) + theta_pi_r_offset  * omega[2])
        else: 
            # Original
            theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + at.dot(beta_rho_s, X), sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
            theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + at.dot(beta_rho_r, X), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
            theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + at.dot(beta_pi_r, X),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

        # psi: True M protein at time 0
        # 1) Normal. Fast convergence, but possibly negative tail 
        if psi_prior=="normal":
            psi = pm.Normal("psi", mu=yi0, sigma=sigma_obs, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        # 2) Lognormal. Works if you give it time to converge
        if psi_prior=="lognormal":
            xi = pm.HalfNormal("xi", sigma=1)
            log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=xi, shape=N_patients)
            psi = pm.Deterministic("psi", np.exp(log_psi))
        # 3) Exact but does not work: 
        #log_psi = pm.Normal("log_psi", mu=np.log(yi0) - np.log( (sigma_obs**2)/(yi0**2) - 1), sigma=np.log( (sigma_obs**2)/(yi0**2) - 1), shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma_obs 
        #psi = pm.Deterministic("psi", np.exp(log_psi))

        # Transformed latent variables 
        rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
        rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
        pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

        # Observation model 
        mu_Y = psi * (pi_r*np.exp(rho_r*t) + (1-pi_r)*np.exp(rho_s*t))

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y)
    # Visualize model
    import graphviz 
    gv = pm.model_to_graphviz(multiple_patients_model)
    gv.render(filename="./plots/posterior_plots/"+name+"_graph_of_model", format="png", view=False)
    # Sample from prior:
    with multiple_patients_model:
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
    plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples_below_"+str(plotlimit_prior)+".png")
    plt.close()
    # All samples: 
    plt.figure()
    az.plot_dist(raveled_Y_true, color="C1", label="observed", bw=3)
    az.plot_dist(raveled_Y_sample, label="simulated", bw=3)
    plt.title("Samples from prior compared to observations")
    plt.xlabel("Y (M protein)")
    plt.ylabel("Frequency")
    plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples.png")
    plt.close()
    # Draw samples from posterior:
    with multiple_patients_model:
        if method == "ADVI":
            inference = pm.ADVI()
            approx = pm.fit(n=50000, method=inference)
            trace = approx.sample(draws=5000)
            idata = trace # Valid?                        
        else: # if method == "HMC"
            # init = "advi", "advi+adapt_diag"
            idata = pm.sample(draws=N_samples, init="advi+adapt_diag", tune=N_tuning, random_seed=42, target_accept=target_accept, max_treedepth=max_treedepth)
    return idata
