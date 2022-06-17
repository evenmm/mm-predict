# Define patient group with parameters Y and covariates X
# Simulate M protein data for each patient
# Estimate Y parameters from the M protein data
# The mapping between X and estimated Y is done in *learn_mapping and *binary_outcome
from utilities import *

# Case 1: 
N_patients = 100
# Two patient groups: Hyperdiploid (A) and Non-hyperdiploid (B)
# They have no history, and all patients get the same treatment for the same amount of days. The treatment is observed through all the interval.
# a) HRD completely resistant, Non-HRD completely sensitive
average_pi_r_HRD = 1
average_pi_r_non_HRD = 0
# b) Both groups partially resistant, but HRD more
#average_pi_r_HRD = 1
#average_pi_r_non_HRD = 0
variance_in_pi_r_both_groups = 0
observation_std_m_protein = 0

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

## Inference
# For inferring both k_1 and growth rates
##               Y_0, pi_r,   g_r,   g_s,  k_1
#lb = np.array([   0,    0,  0.00,  0.00, 0.00])
#ub = np.array([1000,    1,  2.00,  2.00, 2.00])
N_iter = 10000 # Number of independent starting points in parameter estimation

#              Y_0, pi_r,   g_r,   g_s,  k_1,  sigma
lb = np.array([  0,    0,  0.00,   0.00, 0.20]) #, 10e-6])
ub = np.array([100,    1,  0.20,  lb[4], 1.00]) #, 10e4])

# Define patient parameters and measure M protein
# Create Y df for parameters and binary outcomes 
print("Estimating parameters...")
patient_dictionary = {}
Y_parameters = np.array([])
Y_increase_or_not = np.array([])
for training_instance_id in range(N_patients):
    print("Patient", str(training_instance_id))
    if training_instance_id < N_patients/2: # First half of patients are HRD
        expected_pi_r = average_pi_r_HRD
    else:
        expected_pi_r = average_pi_r_non_HRD
    these_parameters = Parameters(Y_0=Y_0_population, pi_r=expected_pi_r, g_r=g_r_population, g_s=g_s_population, k_1=k_1_population, sigma=observation_std_m_protein)
    this_patient = Patient(these_parameters, measurement_times, treatment_history, name=str(training_instance_id))
    patient_dictionary[training_instance_id] = this_patient

    # Estimate parameters: 
    this_estimate = estimate_drug_response_parameters(this_patient, lb, ub, N_iterations=N_iter)
    Y_parameters = np.concatenate((Y_parameters, np.array([this_estimate]))) # training_instance_id is position in Y_parameters
    period_start = end_of_history
    binary_outcome = get_binary_outcome(period_start, this_patient, this_estimate)
    Y_increase_or_not = np.concatenate((Y_increase_or_not, np.array([binary_outcome])))

    # Plot truth and estimates
    plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=this_estimate, PLOT_ESTIMATES=True, plot_title="Simulated patient "+str(training_instance_id), savename="./plots/simulation_plots/patient_"+str(training_instance_id)+".png")

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
ax.set_ylim(bottom=lb[1], top=ub[1])
ax.legend(loc="best")
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
ax.set_ylim(bottom=lb[2], top=ub[2])
ax.legend(loc="best")
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
ax.set_ylim(bottom=lb[3], top=ub[3])
ax.legend(loc="best")
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
ax.set_ylim(bottom=lb[4], top=ub[4])
ax.legend(loc="best")
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
ax.set_ylim(bottom=lb[3]-ub[4], top=0)
ax.legend(loc="best")
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

