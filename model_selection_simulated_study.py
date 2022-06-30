# Define patient group with parameters Y and covariates X
# Simulate M protein data for each patient
# Estimate Y parameters from the M protein data
# The mapping between X and estimated Y is done in *learn_mapping and *binary_outcome
from utilities import *
warnings.simplefilter("ignore")
start_time = time.time()
def get_minval_maxval(list_1, list_2, list_3, start_id, stop_id):
    maxval = max(max(list_1[start_id:stop_id]), max(list_2[start_id:stop_id]), max(list_3[start_id:stop_id]))
    minval = min(min(list_1[start_id:stop_id]), min(list_2[start_id:stop_id]), min(list_3[start_id:stop_id]))
    diff = maxval - minval
    maxval = maxval + 0.1*diff
    minval = minval - 0.1*diff
    return minval, maxval

# Case 1: 
N_patients = 10
N_iter = 1000 # Number of independent starting points in parameter estimation
# Two patient groups: Hyperdiploid (A) and Non-hyperdiploid (B)
# They have no history, and all patients get the same treatment for the same amount of days. The treatment is observed through all the interval.
# a) HRD completely resistant, Non-HRD completely sensitive
average_pi_r_HRD = 1
average_pi_r_non_HRD = 0
# b) Both groups partially resistant, but HRD more
#average_pi_r_HRD = 1
#average_pi_r_non_HRD = 0
variance_in_pi_r_both_groups = 0
observation_std_m_protein = 1 # sigma 

Y_0_population = 50
g_r_population = 0.002
g_s_population = 0.010
k_1_population = 0.015

# Create the covariates X
df_X_covariates = pd.DataFrame(
    {"training_instance_id" : [ii for ii in range(N_patients)],
    "HRD" : [0 for ii in range(N_patients)]}
)
df_X_covariates.loc[(df_X_covariates["training_instance_id"] < N_patients/2), "HRD"] = 1
#df_drugs_and_dates.loc[(df_drugs_and_dates["MMTX_THERAPY"] == drug_name),'drug_id'] = drug_id
#print(df_X_covariates.head(n=N_patients))
print("Average HRD value:", np.mean(df_X_covariates["HRD"].head(n=N_patients)))


# Create patients with M protein under treatment and estimate parameters Y, save them to outcome df
"""
For each training instance id: 
Sample parameters that correspond to covariates
Create a patient and generate M protein 
Estimate parameters and add to Y_parameters df
Add binary outcome to Y_increase_or_not
"""

# M protein measurement settings
days_between_measurements = 180 # every X days
number_of_measurements = 8 # for N*X days
measurement_times = days_between_measurements * np.linspace(0, number_of_measurements, number_of_measurements+1)

# Currently there is no history
treatment_history = np.array([
    Treatment(start=0, end=measurement_times[-1], id=1),
    #Treatment(start=0, end=measurement_times[4], id=1),
    #Treatment(start=measurement_times[4], end=measurement_times[8], id=0),
    ])
end_of_history = 0

# Model 1: exp rho t            (3 parameters: Y0, rho, sigma)
# Model 2: exp t(alpha - k)     (4 parameters: Y0, alpha, K, sigma)
# Model 3: Both together.       (6 parameters: Y0, pi, rho, alpha, K, sigma)

# Bounds for model 1: 1 population, only resistant cells
#                Y_0,  g_r  sigma
lb_1 = np.array([  0, 0.00, 10e-6])
ub_1 = np.array([100, 0.20, 10e4])

# Bounds for model 2: 1 population, only sensitive cells
#                Y_0,   g_s,    k_1,  sigma
lb_2 = np.array([  0,   0.00,  0.20, 10e-6])
ub_2 = np.array([100, lb_2[2], 1.00, 10e4])

# Bounds for model 3: sensitive and resistant population
#                Y_0, pi_r,   g_r,   g_s,     k_1,  sigma
lb_3 = np.array([  0,    0,  0.00,   0.00,   0.20, 10e-6])
ub_3 = np.array([100,    1,  0.20,  lb_3[4], 1.00, 10e4])

bic_penalty_model_1 = len(lb_1)*np.log(number_of_measurements)
bic_penalty_model_2 = len(lb_2)*np.log(number_of_measurements)
bic_penalty_model_3 = len(lb_3)*np.log(number_of_measurements)

print("bic_penalty_model_1:", bic_penalty_model_1)
print("bic_penalty_model_2:", bic_penalty_model_2)
print("bic_penalty_model_3:", bic_penalty_model_3)

# Define patient parameters and measure M protein
# Create Y df for parameters and binary outcomes 
print("Estimating parameters...")
patient_dictionary = {}
Y_parameters = np.array([])
Y_increase_or_not = np.array([])
negative_loglikelihoods_1 = []
negative_loglikelihoods_2 = []
negative_loglikelihoods_3 = []
bic_values_1 = []
bic_values_2 = []
bic_values_3 = []
sigma_estimates_model_1 = []
sigma_estimates_model_2 = []
sigma_estimates_model_3 = []
chosen_models = []
for training_instance_id in range(N_patients):
    print("Patient", str(training_instance_id))
    if training_instance_id < N_patients/2: # First half of patients are HRD
        expected_pi_r = average_pi_r_HRD
    else:
        expected_pi_r = average_pi_r_non_HRD
    these_parameters = Parameters(Y_0=Y_0_population, pi_r=expected_pi_r, g_r=g_r_population, g_s=g_s_population, k_1=k_1_population, sigma=observation_std_m_protein)
    this_patient = Patient(these_parameters, measurement_times, treatment_history, name=str(training_instance_id))
    patient_dictionary[training_instance_id] = this_patient

    # Estimate parameters for model 1
    # Model 1: exp rho t            (2+1=3 parameters: Y0, rho, sigma)
    this_estimate = estimate_drug_response_parameters_any_model(this_patient, lb_1, ub_1, N_iterations=N_iter) #, sigma_noise_std=1
    array_x = np.array([this_estimate.Y_0, this_estimate.g_r, this_estimate.sigma])
    predictions = measure_Mprotein_noiseless(this_estimate, this_patient.measurement_times, this_patient.treatment_history)
    sumofsquares_model_1 = np.sum((this_patient.Mprotein_values - predictions)**2)
    sample_variance_unadjusted_model_1 = sumofsquares_model_1/len(this_patient.measurement_times)
    negative_loglikelihoods_1.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_1))
    #print(negative_loglikelihoods_1[training_instance_id])
    sigma_estimates_model_1.append(array_x[-1])

    # Estimate parameters for model 2
    # Model 2: exp t(alpha - k)     (3+1=4 parameters: Y0, alpha, K, sigma)
    this_estimate = estimate_drug_response_parameters_any_model(this_patient, lb_2, ub_2, N_iterations=N_iter) #, sigma_noise_std=1
    array_x = np.array([this_estimate.Y_0, this_estimate.g_s, this_estimate.k_1, this_estimate.sigma])
    predictions = measure_Mprotein_noiseless(this_estimate, this_patient.measurement_times, this_patient.treatment_history)
    sumofsquares_model_2 = np.sum((this_patient.Mprotein_values - predictions)**2)
    sample_variance_unadjusted_model_2 = sumofsquares_model_2/len(this_patient.measurement_times)
    negative_loglikelihoods_2.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_2))
    #print(negative_loglikelihoods_2[training_instance_id])
    sigma_estimates_model_2.append(array_x[-1])

    # Estimate parameters for model 3
    # Model 3: Both together.       (5+1=6 parameters: Y0, pi, rho, alpha, K, sigma)
    this_estimate = estimate_drug_response_parameters_any_model(this_patient, lb_3, ub_3, N_iterations=N_iter) #, sigma_noise_std=1
    Y_parameters = np.concatenate((Y_parameters, np.array([this_estimate]))) # training_instance_id is position in Y_parameters
    period_start = end_of_history
    binary_outcome = get_binary_outcome(period_start, this_patient, this_estimate)
    Y_increase_or_not = np.concatenate((Y_increase_or_not, np.array([binary_outcome])))
    array_x = np.array([this_estimate.Y_0, this_estimate.pi_r, this_estimate.g_r, this_estimate.g_s, this_estimate.k_1, this_estimate.sigma])
    predictions = measure_Mprotein_noiseless(this_estimate, this_patient.measurement_times, this_patient.treatment_history)
    sumofsquares_model_3 = np.sum((this_patient.Mprotein_values - predictions)**2)
    sample_variance_unadjusted_model_3 = sumofsquares_model_3/len(this_patient.measurement_times)
    negative_loglikelihoods_3.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_3))
    #print(negative_loglikelihoods_3[training_instance_id])
    sigma_estimates_model_3.append(array_x[-1])

    N_observations_this_period = len(this_patient.measurement_times)
    # BIC values
    bic_values_1.append(len(lb_1)*np.log(N_observations_this_period) + 2*negative_loglikelihoods_1[training_instance_id])
    bic_values_2.append(len(lb_2)*np.log(N_observations_this_period) + 2*negative_loglikelihoods_2[training_instance_id])
    bic_values_3.append(len(lb_3)*np.log(N_observations_this_period) + 2*negative_loglikelihoods_3[training_instance_id])
    print("BIC")
    print(bic_values_1[training_instance_id])
    print(bic_values_2[training_instance_id])
    print(bic_values_3[training_instance_id])

    # Model selection (1 2 or 3) by BIC. Then save outcomes
    bic_values_this_patient = bic_values_1[training_instance_id], bic_values_2[training_instance_id], bic_values_3[training_instance_id]
    chosen_models.append(bic_values_this_patient.index(min(bic_values_this_patient)) + 1)
    # Enable selection by BIC, make get_binary_outcome and other learning functions handle "this_estimate" of different lengths, then fix the following instead of after model 3.
    #Y_parameters = np.concatenate((Y_parameters, np.array([this_estimate]))) # training_instance_id is position in Y_parameters
    #period_start = end_of_history
    #binary_outcome = get_binary_outcome(period_start, this_patient, this_estimate)
    #Y_increase_or_not = np.concatenate((Y_increase_or_not, np.array([binary_outcome])))

    # Plot truth and estimates
    #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=this_estimate, PLOT_ESTIMATES=True, plot_title="Simulated patient "+str(training_instance_id), savename="./plots/simulation_plots/patient_"+str(training_instance_id)+".png")
    end_time = time.time()
    time_duration = end_time - start_time
    print("Time elapsed:", time_duration)

print("Models chosen with BIC:\n", [(ii, chosen_models[ii]) for ii in range(len(chosen_models))])

print("Negative loglikelihood:")
print(negative_loglikelihoods_1)
print(negative_loglikelihoods_2)
print(negative_loglikelihoods_3)

print("BIC:")
print(bic_values_1)
print(bic_values_2)
print(bic_values_3)

# Plot estimated sigma values
if N_patients <= 20:
    fig = plt.figure(figsize=(6, 6))
else:
    fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(111)
ax1.axhline(observation_std_m_protein, color="k", label="True sigma", zorder=-5)
ax1.scatter(range(N_patients),sigma_estimates_model_1, color="r", label="Model 1: Only resistant")
ax1.scatter(range(N_patients),sigma_estimates_model_2, color="b", label="Model 2: Only sensitive")
ax1.scatter(range(N_patients),sigma_estimates_model_3, color="g", label="Model 3: Full model")
ax1.set_yscale('log')
observation_std_m_protein
ax1.set_xlabel("Patient")
ax1.set_ylabel("Estimated sigma")
ax1.set_title("Estimated sigma for the three models")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/model_comparison_sigma_estimates_sigma_"+str(observation_std_m_protein)+"_"+str(N_iter)+"_iterations_"+str(N_patients)+"_patients.png")
plt.show()
plt.close()

# Plot loglikelihoods
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
minval, maxval = get_minval_maxval(negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, 0, int(N_patients/2))
for ii in range(0,int(N_patients/2)):
    logl_values_this_patient = [negative_loglikelihoods_2[ii], negative_loglikelihoods_1[ii], negative_loglikelihoods_3[ii]]
    ax1.plot([1,2,3], logl_values_this_patient, marker='x', linestyle='-', linewidth=1)
    min_position = logl_values_this_patient.index(min(logl_values_this_patient))
    ax1.scatter([1+min_position], [logl_values_this_patient[min_position]], s=60)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(["Only sensitive", "Only resistant", "Full model"])
ax1.set_yscale('log')
ax1.set_ylim((minval, maxval))
ax1.set_ylabel("Negative loglikelihood")
ax1.set_title("Resistant patients")

ax2 = fig.add_subplot(122)
minval, maxval = get_minval_maxval(negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, int(N_patients/2), N_patients)
for ii in range(int(N_patients/2),N_patients):
    logl_values_this_patient = [negative_loglikelihoods_1[ii], negative_loglikelihoods_2[ii], negative_loglikelihoods_3[ii]]
    ax2.plot([1,2,3], logl_values_this_patient, marker='x', linestyle='-', linewidth=1)
    min_position = logl_values_this_patient.index(min(logl_values_this_patient))
    ax2.scatter([1+min_position], [logl_values_this_patient[min_position]], s=60)
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(["Only resistant", "Only sensitive", "Full model"])
ax2.set_yscale('log')
ax2.set_ylim((minval, maxval))
ax2.set_ylabel("Negative loglikelihood")
ax2.set_title("Sensitive patients")
plt.suptitle("Negative loglikelihood")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/model_comparison_loglikelihood_sigma_"+str(observation_std_m_protein)+"_"+str(N_iter)+"_iterations_"+str(N_patients)+"_patients.png")
#plt.show()
plt.close()

# Plot BIC
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
minval, maxval = get_minval_maxval(bic_values_1, bic_values_2, bic_values_3, 0, int(N_patients/2))
for ii in range(0,int(N_patients/2)):
    logl_values_this_patient = [bic_values_2[ii], bic_values_1[ii], bic_values_3[ii]]
    ax1.plot([1,2,3], logl_values_this_patient, marker='x', linestyle='-', linewidth=1)
    min_position = logl_values_this_patient.index(min(logl_values_this_patient))
    ax1.scatter([1+min_position], [logl_values_this_patient[min_position]], s=60)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(["Only sensitive", "Only resistant", "Full model"])
ax1.set_yscale('log')
ax1.set_ylim((minval, maxval))
ax1.set_ylabel("BIC")
ax1.set_title("Resistant patients")

ax2 = fig.add_subplot(122)
minval, maxval = get_minval_maxval(bic_values_1, bic_values_2, bic_values_3, int(N_patients/2), N_patients)
for ii in range(int(N_patients/2),N_patients):
    logl_values_this_patient = [bic_values_1[ii], bic_values_2[ii], bic_values_3[ii]]
    ax2.plot([1,2,3], logl_values_this_patient, marker='x', linestyle='-', linewidth=1)
    min_position = logl_values_this_patient.index(min(logl_values_this_patient))
    ax2.scatter([1+min_position], [logl_values_this_patient[min_position]], s=60)
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(["Only resistant", "Only sensitive", "Full model"])
ax2.set_yscale('log')
ax2.set_ylim((minval, maxval))
ax2.set_ylabel("BIC")
ax2.set_title("Sensitive patients")

plt.suptitle("BIC")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/model_comparison_bic_values_sigma_"+str(observation_std_m_protein)+"_"+str(N_iter)+"_iterations_"+str(N_patients)+"_patients.png")
plt.show()
plt.close()

"""
# Plot parameter estimates 
# pi_r
pi_r_estimates = [elem.pi_r for elem in Y_parameters]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([0,N_patients/2-1], [1,1], color='k', marker='', linestyle='-', linewidth=1, label="Truth", zorder=5)
plt.plot([N_patients/2, N_patients-1], [0,0], color='k', marker='', linestyle='-', linewidth=1, zorder=5)
plt.plot(range(N_patients), pi_r_estimates, marker="x", linestyle="", c="k", label="Step 1: Estimate", zorder=10)
#plt.scatter(range(N_patients), pi_r_estimates, c="navy", s=s, edgecolor="black", label="Step 1: Estimate", zorder=10)
plt.xlabel("Patients")
plt.ylabel("pi_R: Resistant cell fraction")
plt.title("Estimated pi_R by least squares")
ax.set_ylim(bottom=lb_3[1], top=ub_3[1])
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/estimated_pi_R.png")
plt.show()

# g_r
g_r_estimates = [elem.g_r for elem in Y_parameters]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([0,N_patients-1], [g_r_population, g_r_population], color='k', marker='', linestyle='-', linewidth=1, label="Truth", zorder=5)
plt.plot(range(N_patients), g_r_estimates, marker="x", linestyle="", c="k", label="Step 1: Estimate", zorder=10)
#plt.scatter(range(N_patients), g_r_estimates, c="navy", s=s, edgecolor="black", label="Step 1: Estimate", zorder=10)
plt.xlabel("Patients")
plt.ylabel("g_r: Resistant cell fraction")
plt.title("Estimated g_r by least squares")
ax.set_ylim(bottom=lb_3[2], top=ub_3[2])
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/estimated_g_r.png")
plt.show()

# g_s
g_s_estimates = [elem.g_s for elem in Y_parameters]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([0,N_patients-1], [g_s_population, g_s_population], color='k', marker='', linestyle='-', linewidth=1, label="Truth", zorder=5)
plt.plot(range(N_patients), g_s_estimates, marker="x", linestyle="", c="k", label="Step 1: Estimate", zorder=10)
#plt.scatter(range(N_patients), g_s_estimates, c="navy", s=s, edgecolor="black", label="Step 1: Estimate", zorder=10)
plt.xlabel("Patients")
plt.ylabel("g_s: Resistant cell fraction")
plt.title("Estimated g_s by least squares")
ax.set_ylim(bottom=lb_3[3], top=ub_3[3])
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/estimated_g_s.png")
plt.show()

# k_1
k_1_estimates = [elem.k_1 for elem in Y_parameters]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([0,N_patients-1], [k_1_population, k_1_population], color='k', marker='', linestyle='-', linewidth=1, label="Truth", zorder=5)
plt.plot(range(N_patients), k_1_estimates, marker="x", linestyle="", c="k", label="Step 1: Estimate", zorder=10)
#plt.scatter(range(N_patients), k_1_estimates, c="navy", s=s, edgecolor="black", label="Step 1: Estimate", zorder=10)
plt.xlabel("Patients")
plt.ylabel("k_1: Resistant cell fraction")
plt.title("Estimated k_1 by least squares")
ax.set_ylim(bottom=lb_3[4], top=ub_3[4])
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/estimated_k_1.png")
plt.show()

# g_s - k_1
g_s_estimates = np.array([elem.g_s for elem in Y_parameters])
k_1_estimates = np.array([elem.k_1 for elem in Y_parameters])
g_s_K_estimates = g_s_estimates - k_1_estimates
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([0,N_patients-1], [g_s_population - k_1_population, g_s_population - k_1_population], color='k', marker='', linestyle='-', linewidth=1, label="Truth", zorder=5)
plt.plot(range(N_patients), g_s_K_estimates, marker="x", linestyle="", c="k", label="Step 1: Estimate", zorder=10)
#plt.scatter(range(N_patients), g_s_K_estimates, c="navy", s=s, edgecolor="black", label="Step 1: Estimate", zorder=10)
plt.xlabel("Patients")
plt.ylabel("(g_s - K): Growth rate of sensitive cells under treatment")
plt.title("Estimated (g_s - K) by least squares")
ax.set_ylim(bottom=lb_3[3]-ub_3[4], top=0)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/simulation_plots/estimated_g_s-K.png")
plt.show()

# Save variables 
picklefile = open('./binaries_and_pickles/df_X_covariates_simulation_study', 'wb')
pickle.dump(np.array(df_X_covariates), picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_parameters_simulation_study', 'wb')
pickle.dump(np.array(Y_parameters), picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_increase_or_not_simulation_study', 'wb')
pickle.dump(Y_increase_or_not, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/patient_dictionary_simulation_study', 'wb')
pickle.dump(patient_dictionary, picklefile)
picklefile.close()
"""