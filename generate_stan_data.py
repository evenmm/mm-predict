# Define patient group with covariates X, effects alpha and beta, then generate data Y
from utilities import *
warnings.simplefilter("ignore")
start_time = time.time()

# Patients have no history, they all get the same treatment for the same amount of days. The treatment is observed through the entire interval.
N_patients = 300
observation_std_m_protein = 0.2 # sigma 
omega = [0.01, 0.01, 0.01]

# Covariates X are centered around zero, making alpha the populations mean 
x1_mean = 0
x1_std = 0.5
df_X_covariates = pd.DataFrame(
    {"training_instance_id" : [ii for ii in range(N_patients)],
    "x1" : [np.random.normal(x1_mean, x1_std) for ii in range(N_patients)]}
    #"observed_psi_0" : [np.random.normal(psi_population, observation_std_m_protein) for ii in range(N_patients)]} # If this is available. Then set the observed value to this further down. 
)
#print(df_X_covariates.head(n=15))

# Desired population_means = alpha + mean_x * beta
rho_s_population = -0.005
rho_r_population = 0.001
pi_r_population = 0.4
psi_population = 50

# We get alpha by taking the inverse transform of the population means 
alpha_rho_s = np.log(-rho_s_population)
alpha_rho_r = np.log(rho_r_population)
alpha_pi_r = np.log(pi_r_population/(1-pi_r_population)) # logit

alpha = np.array([alpha_rho_s, alpha_rho_r, alpha_pi_r])
beta = np.array([[1], # X in range -1 to 1 gives rates between 0.0007 and 0.005
                 [1], # X in range -1 to 1 gives rates between 0.0018 and 0.0135
                 [1], # X in range -1 to 1 gives pi_r between 0.73 and 0.26
])
omega_1 = 0.001
omega_2 = 0.001
omega_3 = 0.01
omega_4 = 0.1
omega = np.array([omega_1, omega_2, omega_3])

# M protein measurement settings
days_between_measurements = 30 # every X days
number_of_measurements = 50 # for N*X days
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements, number_of_measurements+1)

# Currently there is no history
treatment_history = np.array([
    Treatment(start=0, end=measurement_times[-1], id=1),
    #Treatment(start=0, end=measurement_times[4], id=1),
    #Treatment(start=measurement_times[4], end=measurement_times[8], id=0),
    ])
end_of_history = 0

print("Generating parameters and data...")
patient_dictionary = {}
true_parameters = [[] for i in range(N_patients)]
true_rho_s = np.zeros(N_patients)
true_rho_r = np.zeros(N_patients)
true_pi_r = np.zeros(N_patients)
true_psi = np.zeros(N_patients)
for training_instance_id in range(N_patients):
    print("\nPatient ", training_instance_id)
    x1_as_panda_slize = df_X_covariates.loc[(df_X_covariates["training_instance_id"] == training_instance_id), "x1"]
    x1 = np.array(x1_as_panda_slize)

    # Calculate parameters for this patient using deterministic alpha and beta
    expected_theta_1_patient_i = alpha_rho_s + beta[0]*x1
    expected_theta_2_patient_i = alpha_rho_r + beta[1]*x1
    expected_theta_3_patient_i = alpha_pi_r + beta[2]*x1
    expected_theta_4_patient_i = np.log(psi_population)

    # Optional variability in thetas. For now, eta = 0 in data generation
    theta_1_patient_i = expected_theta_1_patient_i # np.random.normal(expected_theta_1_patient_i, omega_1)
    theta_2_patient_i = expected_theta_2_patient_i # np.random.normal(expected_theta_2_patient_i, omega_2)
    theta_3_patient_i = expected_theta_3_patient_i # np.random.normal(expected_theta_3_patient_i, omega_3)
    theta_4_patient_i = np.random.normal(expected_theta_4_patient_i, omega_4)

    # Transform thetas into parameters
    rho_s_patient_i = - np.exp(theta_1_patient_i)
    rho_r_patient_i = np.exp(theta_2_patient_i)
    pi_r_patient_i = 1/(1+np.exp(-theta_3_patient_i)) # sigmoid 
    psi_patient_i = np.exp(theta_4_patient_i)
    print("x1   ;", x1)
    print("rho_s:", rho_s_patient_i)
    print("rho_r:", rho_r_patient_i)
    print("pi_r :", pi_r_patient_i)
    print("psi  :", psi_patient_i)

    true_rho_s[training_instance_id] = rho_s_patient_i
    true_rho_r[training_instance_id] = rho_r_patient_i
    true_pi_r[training_instance_id] = pi_r_patient_i
    true_psi[training_instance_id] = psi_patient_i

    these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=observation_std_m_protein)
    this_patient = Patient(these_parameters, measurement_times, treatment_history, name=str(training_instance_id))
    patient_dictionary[training_instance_id] = this_patient
    true_parameters[training_instance_id] = these_parameters
    plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id))

# x is stored in df_X_covariates
print(df_X_covariates.head(n=15))
picklefile = open('./binaries_and_pickles/Bayesian_df_X_covariates_simulation_study', 'wb')
pickle.dump(np.array(df_X_covariates), picklefile)
picklefile.close()
print("Average x1 value:", np.mean(df_X_covariates["x1"].head(n=N_patients)))

# y and times are stored in the patient dictionary 
picklefile = open('./binaries_and_pickles/Bayesian_patient_dictionary_simulation_study', 'wb')
pickle.dump(patient_dictionary, picklefile)
picklefile.close()

# true rho_s
picklefile = open('./binaries_and_pickles/Bayesian_true_rho_s_simulation_study', 'wb')
pickle.dump(np.array(true_rho_s), picklefile)
picklefile.close()
# true rho_r
picklefile = open('./binaries_and_pickles/Bayesian_true_rho_r_simulation_study', 'wb')
pickle.dump(np.array(true_rho_r), picklefile)
picklefile.close()
# true pi_r
picklefile = open('./binaries_and_pickles/Bayesian_true_pi_r_simulation_study', 'wb')
pickle.dump(np.array(true_pi_r), picklefile)
picklefile.close()
# true psi_r
picklefile = open('./binaries_and_pickles/Bayesian_true_psi_simulation_study', 'wb')
pickle.dump(np.array(true_psi), picklefile)
picklefile.close()
# true parameters
picklefile = open('./binaries_and_pickles/Bayesian_true_parameters_simulation_study', 'wb')
pickle.dump(np.array(true_parameters), picklefile)
picklefile.close()
