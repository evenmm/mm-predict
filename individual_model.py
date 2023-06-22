from utilities import *
import arviz as az
import pymc as pm
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
##############################
# Function argument shapes: 
# patient dictionary contains N_patients patients

def individual_model(patient_dictionary, name, N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, psi_prior="lognormal", FUNNEL_REPARAMETRIZATION=False, method="HMC"):
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
    assert len(Y_flat_no_nans) == len(t_flat_no_nans), "Number of measurements and number of times must be equal"
    assert len(Y_flat_no_nans) > 0, "No measurements found"
    assert min(Y_flat_no_nans) >= 0, "M protein values must be non-negative"
    assert min(t_flat_no_nans) >= 0, "Time values must be non-negative"

    N_patients = len(patient_dictionary)
    yi0 = np.zeros(N_patients)
    for ii in range(N_patients):
        yi0[ii] = max(patient_dictionary[ii].Mprotein_values[0], 1e-5)
    assert min(yi0) > 0, "Initial M protein values yi0 must be positive due to lognormal prior on psi"

    if psi_prior not in ["lognormal", "normal"]:
        print("Unknown prior option specified for psi; Using 'lognormal' prior")
        psi_prior = "lognormal"

    with pm.Model() as individual_model:
        # Observation noise (std)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

        # alpha
        alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)

        # Latent variables theta
        omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
        if FUNNEL_REPARAMETRIZATION == True: 
            # Reparametrized to escape/explore the funnel of Hell (https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/):
            theta_rho_s_offset = pm.Normal('theta_rho_s_offset', mu=0, sigma=1, shape=N_patients)
            theta_rho_r_offset = pm.Normal('theta_rho_r_offset', mu=0, sigma=1, shape=N_patients)
            theta_pi_r_offset  = pm.Normal('theta_pi_r_offset',  mu=0, sigma=1, shape=N_patients)
            theta_rho_s = pm.Deterministic("theta_rho_s", alpha[0] + theta_rho_s_offset * omega[0])
            theta_rho_r = pm.Deterministic("theta_rho_r", alpha[1] + theta_rho_r_offset * omega[1])
            theta_pi_r  = pm.Deterministic("theta_pi_r",  alpha[2] + theta_pi_r_offset  * omega[2])
        else: 
            # Original
            theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0], sigma=omega[0])
            theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1], sigma=omega[1])
            theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2], sdigma=omega[2])

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

        mu_Y = psi[group_id] * (pi_r[group_id]*np.exp(rho_r[group_id]*t_flat_no_nans) + (1-pi_r[group_id])*np.exp(rho_s[group_id]*t_flat_no_nans))

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y_flat_no_nans)
    return individual_model
