from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import aesara.tensor as at

import theano.tensor as tt

# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
##############################
# Function argument shapes: 
# X is an (N_patients, P) shaped pandas dataframe
# patient dictionary contains N_patients patients in the same order as X

def construct_nn(ann_input, ann_output):
    X_train, Y_train = ann_input, ann_output
    n_hidden = 5

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden) #.astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden) #.astype(floatX)
    init_out = np.random.randn(n_hidden) #.astype(floatX)

    with pm.Model() as neural_network:
        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data
        # It's still the same thing, but we can later change the values of the shared variable
        # (to switch in the test-data later) and pymc3 will just use the new data.
        # Kind-of like a pointer we can redirect.
        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", Y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(X.shape[1], n_hidden), testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden,), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            act_out,
            observed=ann_output,
            total_size=Y_train.shape[0],  # IMPORTANT for minibatches
        )
    return neural_network

def sample_from_bayesian_neural_net(X, patient_dictionary, name, N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, psi_prior="lognormal"):
    N_patients, P = X.shape
    P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    X_not_transformed = X.copy()
    X = X.T
    Y = np.transpose(np.array([patient.Mprotein_values for _, patient in patient_dictionary.items()]))
    t = np.transpose(np.array([patient.measurement_times for _, patient in patient_dictionary.items()]))
    yi0 = np.maximum(1e-5, np.array([patient.Mprotein_values[0] for _, patient in patient_dictionary.items()]))
    if psi_prior not in ["lognormal", "normal"]:
        print("Unknown prior option specified for psi; Using 'lognormal' prior")
        psi_prior = "lognormal"

    # Neural network
    neural_network = construct_nn(X, Y)

    with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as multiple_patients_model:
        # Observation noise (std)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # alpha
        alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)


        tt.nnet.relu
        

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
        # 1) Normal. Fast convergence, but possibly negative tail 
        if psi_prior=="normal":
            psi = pm.Normal("psi", mu=yi0, sigma=sigma, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        # 2) Lognormal. Works if you give it time to converge
        if psi_prior=="lognormal":
            xi = pm.HalfNormal("xi", sigma=1)
            log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=xi, shape=N_patients)
            psi = pm.Deterministic("psi", np.exp(log_psi))
        # 3) Exact but does not work: 
        #log_psi = pm.Normal("log_psi", mu=np.log(yi0) - np.log( (sigma**2)/(yi0**2) - 1), sigma=np.log( (sigma**2)/(yi0**2) - 1), shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        #psi = pm.Deterministic("psi", np.exp(log_psi))

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
    plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples.png")
    #plt.show()
    plt.close()
    # Draw samples from posterior:
    with multiple_patients_model:
        idata = pm.sample(N_samples, tune=N_tuning, random_seed=42, target_accept=target_accept, max_treedepth=max_treedepth)
    return idata
