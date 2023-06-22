from utilities import *

# Initialize random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
SAVEDIR = "./plots/individual_estimates/"

######### Generate data #########
true_sigma_obs = 1
RANDOM_EFFECTS = True
RANDOM_EFFECTS_TEST = False
N_patients = 50
N_patients_test = 30
CI_with_obs_noise = False
PLOT_RESISTANT = True
P = 5 # Number of covariates
true_omega = np.array([0.10, 0.05, 0.20])
test_seed = 23
M_number_of_measurements = 7
y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
true_omega_for_psi = 0.1
max_time = 1200 #3000 #1500
days_between_measurements = int(max_time/M_number_of_measurements)
measurement_times = days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

name = "simdata_individual"+"_"+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)

X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_3_pi_rho, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS)
#rho_s_population = -0.005
#rho_r_population = 0.001
#pi_r_population = 0.3

# Visualize parameter dependancy on covariates 
plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r)

###########################################
# Estimate parameters indidivually
rho_s, rho_r, pi_r = estimate_individual_parameters(patient_dictionary, N_iter=10_000, SAVEDIR="./plots/", plotting=False)

# plot estimates and covariates
plot_parameters_and_covariates(SAVEDIR, name, X, rho_s, rho_r, pi_r, covariates=["Covariate 1", "Covariate 2", "Covariate 3"])
