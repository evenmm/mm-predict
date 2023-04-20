from utilities import *
from linear_model import *

# Initialize random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
#SAVEDIR = "/data/evenmm/plots/"
SAVEDIR = "./plots/Bayesian_estimates_simdata_linearmodel/"

script_index = int(sys.argv[1]) 

# Settings
if int(script_index % 3) == 0:
    true_sigma_obs = 0
elif int(script_index % 3) == 1:
    true_sigma_obs = 2.5
elif int(script_index % 3) == 2:
    true_sigma_obs = 5

if script_index >= 3:
    RANDOM_EFFECTS = True
else: 
    RANDOM_EFFECTS = False

RANDOM_EFFECTS_TEST = False

N_patients = 150
psi_prior="lognormal"
N_samples = 1000
N_tuning = 1000
target_accept = 0.99
CI_with_obs_noise = True
FUNNEL_REPARAMETRIZATION = False
MODEL_RANDOM_EFFECTS = True
N_HIDDEN = 2
P = 3 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
true_omega = np.array([0.10, 0.05, 0.20])

M_number_of_measurements = 5
y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
true_omega_for_psi = 0.1

max_time = 1200 #3000 #1500
days_between_measurements = int(max_time/M_number_of_measurements)
measurement_times = days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])
name = "simdata_lin_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_FUNNEL_"+str(FUNNEL_REPARAMETRIZATION)+"_RNDM_EFFECTS_"+str(RANDOM_EFFECTS)
print("Running "+name)

X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s = generate_simulated_patients(deepcopy(measurement_times), deepcopy(treatment_history), true_sigma_obs, N_patients, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS)

# Introduce nan values: 
WITH_MISSING_VALUES = False
#patient_dictionary[0].Mprotein_values[-1] = np.nan
#patient_dictionary[0].measurement_times[-1] = np.nan
patient_dictionary[0].Mprotein_values = patient_dictionary[0].Mprotein_values[0:-1]
patient_dictionary[0].measurement_times = patient_dictionary[0].measurement_times[0:-1]
if WITH_MISSING_VALUES:
    new_pd = {}
    for key, old_patient in patient_dictionary.items(): 
        new_patient = deepcopy(old_patient)
        stop_index = np.random.randint(low=3, high=len(old_patient.Mprotein_values)) # Minimum 3 M protein measurements 
        new_patient.Mprotein_values[stop_index:] = np.nan
        new_patient.measurement_times[stop_index:] = np.nan
        new_pd[key] = new_patient
    del patient_dictionary
    patient_dictionary = new_pd
    iii = 0
    for key, patient in patient_dictionary.items():
        mm = patient.Mprotein_values
        tt = patient.measurement_times
        if iii < 10:
            print(patient.Mprotein_values)
            print(patient.measurement_times)
            iii = iii + 1

# Visualize parameter dependancy on covariates 
VISZ = False
if VISZ:
    color_array = X["Covariate 2"].to_numpy()

    fig, ax = plt.subplots()
    ax.set_title("expected_theta_1 depends on covariates 1 and 2")
    points = ax.scatter(X["Covariate 1"], expected_theta_1, c=color_array, cmap="plasma")
    ax.set_xlabel("covariate 1")
    ax.set_ylabel("expected_theta_1")
    cbar = fig.colorbar(points)
    cbar.set_label('covariate 2', rotation=90)
    plt.savefig(SAVEDIR+"effects_1_"+name+".pdf", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("true_theta_rho_s depends on covariates 1 and 2")
    points = ax.scatter(X["Covariate 1"], true_theta_rho_s, c=color_array, cmap="plasma")
    ax.set_xlabel("covariate 1")
    ax.set_ylabel("true_theta_rho_s")
    cbar = fig.colorbar(points)
    cbar.set_label('covariate 2', rotation=90)
    plt.savefig(SAVEDIR+"effects_2_"+name+".pdf", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("true_rho_s depends on covariates 1 and 2")
    points = ax.scatter(X["Covariate 1"], true_rho_s, c=color_array, cmap="plasma")
    ax.set_xlabel("covariate 1")
    ax.set_ylabel("true_rho_s")
    cbar = fig.colorbar(points)
    cbar.set_label('covariate 2', rotation=90)
    plt.savefig(SAVEDIR+"effects_3_"+name+".pdf", dpi=300)
    plt.close()

# Sample from full model
lin_model = linear_model(X, patient_dictionary, name, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION)
# Draw samples from posterior:
with lin_model:
    """
    print("------------------- INDEPENDENT ADVI -------------------")
    xxxxxx = 8
    advi = pm.ADVI()
    tracker = pm.callbacks.Tracker(
        mean=advi.approx.mean.eval,  # callable that returns mean
        std=advi.approx.std.eval,  # callable that returns std
    )

    approx = advi.fit(xxxxxx, obj_optimizer=pm.adadelta(), callbacks=[tracker])
    advi_trace = approx.sample(1000)
    advi_dict = advi_trace.to_dict()
    posterior_dict = advi_dict["posterior"]
    #for key, value in posterior_dict.items():
    #    print(key)
    #    print(np.median(value[0], axis=0))
    median_dict = {key : np.median(value[0], axis=0) for key, value in posterior_dict.items()}
    #for key, value in median_dict.items():
    #    print(key)
    #    print(value)
    
    #mean_field = pm.fit(method="advi", obj_optimizer=pm.adadelta())
    #post_mean = mean_field.mean.eval()
    #print(post_mean.posterior)
    """


    """
    #approx = advi.fit(xxxxxx, callbacks=[tracker])
    #fig = plt.figure(figsize=(16, 9))
    #mu_ax = fig.add_subplot(221)
    #std_ax = fig.add_subplot(222)
    #hist_ax = fig.add_subplot(212)
    #mu_ax.plot(tracker["mean"])
    #mu_ax.set_title("Mean track")
    #std_ax.plot(tracker["std"])
    #std_ax.set_title("Std track")
    #hist_ax.plot(advi.hist)
    #hist_ax.set_title("Negative ELBO track")
    #hist_ax.set_yscale("log")
    #plt.savefig(SAVEDIR+"0_plain_advi_trace.pdf", dpi=300)
    #plt.show()
    """
    print("-------------------SAMPLING-------------------")
    idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=60000, random_seed=42, target_accept=target_accept)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="jitter+adapt_diag", random_seed=42, target_accept=target_accept)
    # Old. This gives NaN, and has loss bar (Average loss)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="jitter+adapt_diag_grad", random_seed=42, target_accept=target_accept)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", random_seed=42, target_accept=target_accept)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="jitter+adapt_diag", random_seed=42, target_accept=target_accept)

    #step = pm.NUTS
    ## Uses the init method even though init values are provided. Maybe I am mistaken about how initvals work. Are they just the init values of the initialization? Can we not just sample starting from there? 
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", initvals=median_dict, random_seed=42, target_accept=target_accept)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="adapt_full", initvals=median_dict, random_seed=42, target_accept=target_accept)
    
    ### This one autostarts the jitter+adapt_diag algorithm 
    ##idata = pm.sample(draws=N_samples, tune=N_tuning, initvals=median_dict, random_seed=42, target_accept=target_accept) 
    ### Slow sampling and does not state if it is using initvals. Slow. 
    ##idata = pm.sample(draws=N_samples, tune=N_tuning, init="adapt_full", initvals=median_dict, random_seed=42, target_accept=target_accept) 

    ## old, with bad advi initialization: 
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", random_seed=42, target_accept=target_accept)
    # New, with init_vals isntead of init: 
    #idata = pm.sample(draws=N_samples, tune=N_tuning, initvals=, random_seed=42, target_accept=target_accept)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="adapt_full", random_seed=42, target_accept=target_accept)
    #idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi", random_seed=42, target_accept=target_accept)
# This is an xArray: https://docs.xarray.dev/en/v2022.11.0/user-guide/data-structures.html
print("Done sampling")

picklefile = open('./binaries_and_pickles/idata'+name, 'wb')
pickle.dump(idata, picklefile)
picklefile.close()

quasi_geweke_test(idata, model="linear", first=0.1, last=0.5)

##print("Plotting posterior/trace plots")
##plot_posterior_traces(idata, SAVEDIR, name, psi_prior, model="linear")

# Generate test patients
N_patients_test = 20
test_seed = 23
X_test, patient_dictionary_test, parameter_dictionary_test, expected_theta_1_test, true_theta_rho_s_test, true_rho_s_test = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_test, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=test_seed, RANDOM_EFFECTS=RANDOM_EFFECTS_TEST)
print("Done generating test patients")

plot_all_credible_intervals(idata, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name, y_resolution, model="linear", parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise)

# Checking that the X matches the observations and the precictions 
print("Checking that the X matches the observations and the precictions")
expected_theta_1, expected_theta_2, expected_theta_3 = get_expected_theta_from_X_2(X)
true_theta_rho_s = expected_theta_1
true_theta_rho_r = expected_theta_2
true_theta_pi_r  = expected_theta_3
psi_population = 50
true_theta_psi = np.random.normal(np.log(psi_population), true_omega_for_psi, size=N_patients)
true_rho_s = - np.exp(true_theta_rho_s)
true_rho_r = np.exp(true_theta_rho_r)
true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
true_psi = np.exp(true_theta_psi)
for ttt in [0,1]:
    print(X_test.loc[ttt,:])
    print("\n", )
    print(ttt)
    print(true_rho_s[ttt])
    print(true_rho_r[ttt])
    print(true_pi_r[ttt])
