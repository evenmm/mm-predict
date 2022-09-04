# Learn mapping simulation study 3
# Fit the linear regression to the training cases to learn the effect of the simulated binary variable HRD on each of the parameters pi, rho, alpha and K 
from utilities import *
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
warnings.simplefilter("ignore")
start_time = time.time()

# Load saved values
picklefile = open('./binaries_and_pickles/df_X_covariates_simulation_study_3', 'rb')
df_X_covariates = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_parameters_simulation_study_3', 'rb')
Y_parameters = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_increase_or_not_simulation_study_3', 'rb')
Y_increase_or_not = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/patient_dictionary_simulation_study_3', 'rb')
patient_dictionary = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/model_choice_simulation_study_3', 'rb')
model_choice = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/chosen_estimates_simulation_study_3', 'rb')
chosen_estimates = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/true_parameters_simulation_study_3', 'rb')
true_parameters = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/covariate_names_simulation_study_3', 'rb')
covariate_names = pickle.load(picklefile)
picklefile.close()

# How many chosen for each model type: 
print("How many times was model 1 chosen:\n", sum([elem == 1 for elem in model_choice]))
print("How many times was model 2 chosen:\n", sum([elem == 2 for elem in model_choice]))
print("How many times was model 3 chosen:\n", sum([elem == 3 for elem in model_choice]))

# Choose variables to include 
print(df_X_covariates)
X = df_X_covariates[:,1:]
gene_expression_X = df_X_covariates[:,2]
print(X)
average_covariates = np.mean(X,axis=0)

N_patients = len(model_choice)
# Get the parameters in array form 
pi_r_estimates = np.array([param_object.pi_r for param_object in chosen_estimates])
g_r_estimates = np.array([param_object.g_r for param_object in chosen_estimates])
g_s_estimates = np.array([param_object.g_s for param_object in chosen_estimates])
k_1_estimates = np.array([param_object.k_1 for param_object in chosen_estimates])
g_s_minus_K_estimates = np.array([g_s_estimates[ii] - k_1_estimates[ii] for ii in range(len(k_1_estimates))])

true_pi_r_values = np.array([param_object.pi_r for param_object in true_parameters])
true_g_r_values = np.array([param_object.g_r for param_object in true_parameters])
true_g_s_values = np.array([param_object.g_s for param_object in true_parameters])
true_k_1_values = np.array([param_object.k_1 for param_object in true_parameters])
true_g_s_minus_K_values = np.array([true_g_s_values[ii] - true_k_1_values[ii] for ii in range(len(true_k_1_values))])

"""
# Compare truth with estimate
fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_pi_r_values, label="True pi")
plt.scatter(range(0, 100*N_patients ,100), true_pi_r_values)
plt.plot(range(0, 100*N_patients ,100), pi_r_estimates, label="Estimated pi")
plt.scatter(range(0, 100*N_patients ,100), pi_r_estimates)
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("Truth and max likelihood estimates\npi_r: Fraction of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_r_values, label="True rho")
plt.scatter(range(0, 100*N_patients ,100), true_g_r_values)
plt.plot(range(0, 100*N_patients ,100), g_r_estimates, label="Estimated rho")
plt.scatter(range(0, 100*N_patients ,100), g_r_estimates)
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("Truth and max likelihood estimates\nrho: growth rate of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_s_minus_K_values, label="True alpha-K")
plt.scatter(range(0, 100*N_patients ,100), true_g_s_minus_K_values)
plt.plot(range(0, 100*N_patients ,100), g_s_minus_K_estimates, label="Estimated alpha-K")
plt.scatter(range(0, 100*N_patients ,100), g_s_minus_K_estimates)
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Days since treatment start")
plt.title("Truth and max likelihood estimates\nalpha-K: adjusted growth rate of sensitive cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K.png")
plt.tight_layout()
#plt.show()
plt.close()

# Fit a linear regression for each identifiable parameter
# ------------ pi_r -------------
y = pi_r_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
yhat_pi_r = model.predict(X)
# summarize prediction
print("Predictions:", yhat_pi_r)
print("slope and intercept:", model.coef_)

# ------------ g_r -------------
y = g_r_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
yhat_g_r = model.predict(X)
# summarize prediction
print("Predictions:", yhat_g_r)
print("slope and intercept:", model.coef_)

# ------------ g_s_minus_K -------------
y = g_s_minus_K_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
yhat_g_s_minus_K = model.predict(X)
# summarize prediction
print("Predictions:", yhat_g_s_minus_K)
print("slope and intercept:", model.coef_)

# Compare truth with estimate
fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_pi_r_values, label="True pi")
plt.scatter(range(0, 100*N_patients ,100), true_pi_r_values)
plt.plot(range(0, 100*N_patients ,100), pi_r_estimates, label="Estimated pi")
plt.scatter(range(0, 100*N_patients ,100), pi_r_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_pi_r, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_pi_r)
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("Linear model\npi_r: Fraction of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_linreg.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_r_values, label="True rho")
plt.scatter(range(0, 100*N_patients ,100), true_g_r_values)
plt.plot(range(0, 100*N_patients ,100), g_r_estimates, label="Estimated rho")
plt.scatter(range(0, 100*N_patients ,100), g_r_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_g_r, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_g_r)
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("Linear model\nrho: growth rate of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_linreg.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_s_minus_K_values, label="True alpha-K")
plt.scatter(range(0, 100*N_patients ,100), true_g_s_minus_K_values)
plt.plot(range(0, 100*N_patients ,100), g_s_minus_K_estimates, label="Estimated alpha-K")
plt.scatter(range(0, 100*N_patients ,100), g_s_minus_K_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_g_s_minus_K, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_g_s_minus_K)
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Days since treatment start")
plt.title("Linear model\nalpha-K: adjusted growth rate of sensitive cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_linreg.png")
plt.tight_layout()
#plt.show()
plt.close()



############################################################################
#                          SVM
############################################################################
# ------------ pi_r -------------
y = pi_r_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = svm.SVR()
# fit model
model.fit(X, y)
# make a prediction
yhat_pi_r = model.predict(X)
# summarize prediction
print("Predictions:", yhat_pi_r)
# ------------ g_r -------------
y = g_r_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = svm.SVR()
# fit model
model.fit(X, y)
# make a prediction
yhat_g_r = model.predict(X)
# summarize prediction
print("Predictions:", yhat_g_r)
# ------------ g_s_minus_K -------------
y = g_s_minus_K_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = svm.SVR()
# fit model
model.fit(X, y)
# make a prediction
yhat_g_s_minus_K = model.predict(X)
# summarize prediction
print("Predictions:", yhat_g_s_minus_K)

# Compare truth with estimate
fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_pi_r_values, label="True pi")
plt.scatter(range(0, 100*N_patients ,100), true_pi_r_values)
plt.plot(range(0, 100*N_patients ,100), pi_r_estimates, label="Estimated pi")
plt.scatter(range(0, 100*N_patients ,100), pi_r_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_pi_r, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_pi_r)
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("SVM\npi_r: Fraction of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_SVM.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_r_values, label="True rho")
plt.scatter(range(0, 100*N_patients ,100), true_g_r_values)
plt.plot(range(0, 100*N_patients ,100), g_r_estimates, label="Estimated rho")
plt.scatter(range(0, 100*N_patients ,100), g_r_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_g_r, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_g_r)
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("SVM\nrho: growth rate of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_SVM.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_s_minus_K_values, label="True alpha-K")
plt.scatter(range(0, 100*N_patients ,100), true_g_s_minus_K_values)
plt.plot(range(0, 100*N_patients ,100), g_s_minus_K_estimates, label="Estimated alpha-K")
plt.scatter(range(0, 100*N_patients ,100), g_s_minus_K_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_g_s_minus_K, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_g_s_minus_K)
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Days since treatment start")
plt.title("SVM\nalpha-K: adjusted growth rate of sensitive cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_SVM.png")
plt.tight_layout()
#plt.show()
plt.close()

############################################################################
#                          Random forest
############################################################################
# ------------ pi_r -------------
y = pi_r_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = RandomForestRegressor(n_estimators=10)
# fit model
model.fit(X, y)
# make a prediction
yhat_pi_r = model.predict(X)
# summarize prediction
print("Predictions:", yhat_pi_r)
# ------------ g_r -------------
y = g_r_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = RandomForestRegressor(n_estimators=10)
# fit model
model.fit(X, y)
# make a prediction
yhat_g_r = model.predict(X)
# summarize prediction
print("Predictions:", yhat_g_r)
# ------------ g_s_minus_K -------------
y = g_s_minus_K_estimates
print(y)
print("X and y:")
print([(X[ii][1], y[ii]) for ii in range(len(y))])
# define model
model = RandomForestRegressor(n_estimators=10)
# fit model
model.fit(X, y)
# make a prediction
yhat_g_s_minus_K = model.predict(X)
# summarize prediction
print("Predictions:", yhat_g_s_minus_K)

# Compare truth with estimate
fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_pi_r_values, label="True pi")
plt.scatter(range(0, 100*N_patients ,100), true_pi_r_values)
plt.plot(range(0, 100*N_patients ,100), pi_r_estimates, label="Estimated pi")
plt.scatter(range(0, 100*N_patients ,100), pi_r_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_pi_r, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_pi_r)
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("Random forest\npi_r: Fraction of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_random_forest.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_r_values, label="True rho")
plt.scatter(range(0, 100*N_patients ,100), true_g_r_values)
plt.plot(range(0, 100*N_patients ,100), g_r_estimates, label="Estimated rho")
plt.scatter(range(0, 100*N_patients ,100), g_r_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_g_r, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_g_r)
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Days since treatment start")
plt.title("Random forest\nrho: growth rate of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_random_forest.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(range(0, 100*N_patients ,100), true_g_s_minus_K_values, label="True alpha-K")
plt.scatter(range(0, 100*N_patients ,100), true_g_s_minus_K_values)
plt.plot(range(0, 100*N_patients ,100), g_s_minus_K_estimates, label="Estimated alpha-K")
plt.scatter(range(0, 100*N_patients ,100), g_s_minus_K_estimates)
plt.plot(range(0, 100*N_patients ,100), yhat_g_s_minus_K, label="Machine learning prediction")
plt.scatter(range(0, 100*N_patients ,100), yhat_g_s_minus_K)
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Days since treatment start")
plt.title("Random forest\nalpha-K: adjusted growth rate of sensitive cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_random_forest.png")
plt.tight_layout()
#plt.show()
plt.close()

def sort_by_values_in_first_list(first_list, second_list, third_list=[]):
    if len(third_list) > 0: 
        joined_list = [[first_list[ii], second_list[ii], third_list[ii]] for ii, elem in enumerate(first_list)]
    else: 
        joined_list = [[first_list[ii], second_list[ii]] for ii, elem in enumerate(first_list)]
    sorted_joined_list = sorted(joined_list, key = lambda x: x[0])
    sorted_first_list = [elem[0] for elem in sorted_joined_list]
    sorted_second_list = [elem[1] for elem in sorted_joined_list]
    if len(third_list) > 0: 
        sorted_third_list = [elem[2] for elem in sorted_joined_list]
        return sorted_first_list, sorted_second_list, sorted_third_list
    return sorted_first_list, sorted_second_list
"""


"""
# Sort true, estimate and covariates by true. Further down we sort by the variable gene_expression_X
# Sort by pi_r, :
true_pi_r_values, pi_r_estimates = sort_by_values_in_first_list(true_pi_r_values, pi_r_estimates)
true_g_r_values, g_r_estimates = sort_by_values_in_first_list(true_g_r_values, g_r_estimates)
true_g_s_minus_K_values, g_s_minus_K_estimates = sort_by_values_in_first_list(true_g_s_minus_K_values, g_s_minus_K_estimates)

# Compare truth with estimates
fig, ax = plt.subplots()
plt.plot(true_pi_r_values, label="True pi")
plt.scatter(range(N_patients),true_pi_r_values)
plt.plot(pi_r_estimates, label="Estimated pi")
plt.scatter(range(N_patients),pi_r_estimates)
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Patients sorted by true value")
plt.title("Sorted\npi_r: Fraction of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_sorted.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(true_g_r_values, label="True rho")
plt.scatter(range(N_patients),true_g_r_values)
plt.plot(g_r_estimates, label="Estimated rho")
plt.scatter(range(N_patients),g_r_estimates)
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Patients sorted by true value")
plt.title("Sorted\nrho: growth rate of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_sorted.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(true_g_s_minus_K_values, label="True alpha-K")
plt.scatter(range(N_patients),true_g_s_minus_K_values)
plt.plot(g_s_minus_K_estimates, label="Estimated alpha-K")
plt.scatter(range(N_patients),g_s_minus_K_estimates)
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Patients sorted by true value")
plt.title("Sorted\nalpha-K: adjusted growth rate of sensitive cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_sorted.png")
plt.tight_layout()
#plt.show()
plt.close()

# Sort true, estimate and covariates by gene_expression_X
# Sort by pi_r, :
sorted_gene_expression_X, true_pi_r_values, pi_r_estimates = sort_by_values_in_first_list(gene_expression_X, true_pi_r_values, pi_r_estimates)
sorted_gene_expression_X, true_g_r_values, g_r_estimates = sort_by_values_in_first_list(gene_expression_X, true_g_r_values, g_r_estimates)
sorted_gene_expression_X, true_g_s_minus_K_values, g_s_minus_K_estimates = sort_by_values_in_first_list(gene_expression_X, true_g_s_minus_K_values, g_s_minus_K_estimates)

# Compare truth with estimates
fig, ax = plt.subplots()
plt.plot(sorted_gene_expression_X, true_pi_r_values, label="True pi")
plt.scatter(sorted_gene_expression_X, true_pi_r_values)
plt.plot(sorted_gene_expression_X, pi_r_estimates, label="Estimated pi")
plt.scatter(sorted_gene_expression_X, pi_r_estimates)
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Gene expression X")
plt.title("Sorted by gene expression\npi_r: Fraction of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_sorted_by_gene_expression.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(sorted_gene_expression_X, true_g_r_values, label="True rho")
plt.scatter(sorted_gene_expression_X, true_g_r_values)
plt.plot(sorted_gene_expression_X, g_r_estimates, label="Estimated rho")
plt.scatter(sorted_gene_expression_X, g_r_estimates)
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Gene expression X")
plt.title("Sorted by gene expression\nrho: growth rate of resistant cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_sorted_by_gene_expression.png")
plt.tight_layout()
#plt.show()
plt.close()

fig, ax = plt.subplots()
plt.plot(sorted_gene_expression_X, true_g_s_minus_K_values, label="True alpha-K")
plt.scatter(sorted_gene_expression_X, true_g_s_minus_K_values)
plt.plot(sorted_gene_expression_X, g_s_minus_K_estimates, label="Estimated alpha-K")
plt.scatter(sorted_gene_expression_X, g_s_minus_K_estimates)
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Gene expression X")
plt.title("Sorted by gene expression\nalpha-K: adjusted growth rate of sensitive cells")
plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_sorted_by_gene_expression.png")
plt.tight_layout()
#plt.show()
plt.close()
"""

def sort_by_values_in_first_list(all_lists): #all_lists = [a, b, c]
    first_list = all_lists[0]
    joined_list = [[single_list[ii] for single_list in all_lists] for ii, _ in enumerate(first_list)]
    sorted_joined_list = sorted(joined_list, key = lambda x: x[0])
    all_sorted_lists = [[elem[ii] for elem in sorted_joined_list] for ii, _ in enumerate(all_lists)]
    return all_sorted_lists

def find_analytic_parameters(X_covariates):
    days_between_measurements = 60 # every X days
    number_of_measurements = 8 # for N*X days
    observation_std_m_protein = 0 # sigma 
    g_r_population = 0.0060
    g_s_population = 0.0100
    k_1_population = 0.0200
    initial_Y_0 = 50
    initial_pi_r = 0.01
    days_before_treatment_of_interest = X_covariates[0]
    gene_expression_X = X_covariates[1]
    max_deviation = np.sqrt(0.8*(k_1_population - g_s_population))
    g_s_patient_i = g_s_population + gene_expression_X**2
    parameters_at_diagnosis = Parameters(Y_0=initial_Y_0, pi_r=initial_pi_r, g_r=g_r_population, g_s=g_s_patient_i, k_1=k_1_population, sigma=observation_std_m_protein)
    g_r_i, g_s_i, k_1_i = parameters_at_diagnosis.g_r, parameters_at_diagnosis.g_s, parameters_at_diagnosis.k_1
    treatment_of_interest_index = -1 # index in full_treatment_history
    # Measurement times under the treatment of interest (All time points are relative to the date of diagnosis)
    measurement_times_under_treatment_of_interest = days_before_treatment_of_interest + days_between_measurements * np.linspace(0, number_of_measurements, number_of_measurements+1)
    # Full history includes final treatment where we measure M protein. One entry here means that there is no history
    full_treatment_history = np.array([
        Treatment(start=0, end=days_before_treatment_of_interest, id=1),
        Treatment(start=days_before_treatment_of_interest, end=measurement_times_under_treatment_of_interest[-1], id=1),
        ])
    treatment_of_interest = full_treatment_history[treatment_of_interest_index]
    treatments_before_treatment_of_interest = full_treatment_history[0:treatment_of_interest_index]

    at_diagnosis_patient = Patient(parameters_at_diagnosis, measurement_times_under_treatment_of_interest, treatment_history=full_treatment_history, name="Test patient")
    duration_of_treatment_of_interest = treatment_of_interest.end - treatment_of_interest.start
    # Parameters at the start of treatment of interest 
    M_protein_at_treatment_of_interest_start, pi_at_treatment_of_interest_start = get_pi_r_after_time_has_passed(params=parameters_at_diagnosis, measurement_times=np.array([treatment_of_interest.start]), treatment_history=treatments_before_treatment_of_interest)
    parameters_at_treatment_of_interest_start = Parameters(Y_0=M_protein_at_treatment_of_interest_start, pi_r=pi_at_treatment_of_interest_start, g_r=g_r_i, g_s=g_s_i, k_1=k_1_i, sigma=observation_std_m_protein)
    return pi_at_treatment_of_interest_start, g_r_i, (g_s_i-k_1_i)

# for each machine learning model: 
#model = LinearRegression()
model = RandomForestRegressor(n_estimators=10)
# ------------ pi_r -------------
pi_r_model = model.fit(X, pi_r_estimates)
yhat_pi_r = pi_r_model.predict(X)

# Checking that we are censoring:
#print([choice in [1,3] for choice in model_choice])
#print(X[[choice in [1,3] for choice in model_choice]])
#print(len(X))
#print(len(X[[choice in [1,2] for choice in model_choice]]))

# They must be numpy arrays
#print([choice in [2,3] for choice in model_choice])
#print(len(g_r_estimates))
#print(len([choice in [2,3] for choice in model_choice]))
#print(g_r_estimates)
#print(g_r_estimates[[choice in [1,3] for choice in model_choice]])

# ------------ g_r -------------
g_r_model = model.fit(X[[choice in [1,3] for choice in model_choice]], g_r_estimates[[choice in [1,3] for choice in model_choice]])
yhat_g_r = g_r_model.predict(X)
# ------------ g_s_minus_K -------------
g_s_minus_K_model = model.fit(X[[choice in [2,3] for choice in model_choice]], g_s_minus_K_estimates[[choice in [2,3] for choice in model_choice]])
yhat_g_s_minus_K = g_s_minus_K_model.predict(X)

print(yhat_pi_r)
print(yhat_g_r)
print(yhat_g_s_minus_K)
print(X)
print(average_covariates)

# For all covariates in X, plot comparisons of (true, estimated and predicted) parameter values for all parameters as a function of that covariate 
for covariate_index in range(len(X[0])):
    covariate_values = X[:,covariate_index]
    cov_name = covariate_names[covariate_index]
    print(covariate_values)
    print(cov_name)

    sorted_covariate_values, sorted_true_pi_r_values, sorted_pi_r_estimates, sorted_yhat_pi_r, sorted_model_choice = sort_by_values_in_first_list([covariate_values, true_pi_r_values, pi_r_estimates, yhat_pi_r, model_choice])
    
    print(sorted_covariate_values)
    print(sorted_true_pi_r_values)
    print(sorted_pi_r_estimates)
    print(sorted_yhat_pi_r)

    sorted_covariate_values, sorted_true_g_r_values, sorted_g_r_estimates, sorted_yhat_g_r, sorted_model_choice = sort_by_values_in_first_list([covariate_values, true_g_r_values, g_r_estimates, yhat_g_r, model_choice])
    print(sorted_covariate_values)
    print(sorted_true_g_r_values)
    print(sorted_g_r_estimates)
    print(sorted_yhat_g_r)

    sorted_covariate_values, sorted_true_g_s_minus_K_values, sorted_g_s_minus_K_estimates, sorted_yhat_g_s_minus_K, sorted_model_choice = sort_by_values_in_first_list([covariate_values, true_g_s_minus_K_values, g_s_minus_K_estimates, yhat_g_s_minus_K, model_choice])

    print(sorted_covariate_values)
    print(sorted_true_g_s_minus_K_values)
    print(sorted_g_s_minus_K_estimates)
    print(sorted_yhat_g_s_minus_K)

    # Plot truth and estimates without predictions 
    fig, ax = plt.subplots()
    plt.plot(sorted_covariate_values, sorted_true_pi_r_values, label="True pi")
    plt.scatter(sorted_covariate_values, sorted_true_pi_r_values)
    plt.plot(sorted_covariate_values, sorted_pi_r_estimates, label="Estimated pi")
    plt.scatter(sorted_covariate_values, sorted_pi_r_estimates)
    plt.legend()
    plt.ylabel("pi_r: Fraction of resistant cells")
    plt.xlabel(cov_name)
    plt.title("pi_r: Fraction of resistant cells\nsorted by "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_sorted_by_"+cov_name+".png")
    plt.tight_layout()
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(sorted_covariate_values, sorted_true_g_r_values, label="True rho")
    plt.scatter(sorted_covariate_values, sorted_true_g_r_values)
    plt.plot(sorted_covariate_values, sorted_g_r_estimates, label="Estimated rho")
    plt.scatter(sorted_covariate_values, sorted_g_r_estimates)
    plt.legend()
    plt.ylabel("rho: growth rate of resistant cells")
    plt.xlabel(cov_name)
    #ax.set_yscale("log")
    plt.title("rho: growth rate of resistant cells\nsorted by "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_sorted_by_"+cov_name+".png")
    plt.tight_layout()
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(sorted_covariate_values, sorted_true_g_s_minus_K_values, label="True alpha-K")
    plt.scatter(sorted_covariate_values, sorted_true_g_s_minus_K_values)
    plt.plot(sorted_covariate_values, sorted_g_s_minus_K_estimates, label="Estimated alpha-K")
    plt.scatter(sorted_covariate_values, sorted_g_s_minus_K_estimates)
    plt.legend()
    plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
    plt.xlabel(cov_name)
    plt.title("alpha-K: adjusted growth rate of sensitive cells\nsorted by "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_sorted_by_"+cov_name+".png")
    plt.tight_layout()
    #plt.show()
    plt.close()

    # Plot truth, estimates and prediction 
    fig, ax = plt.subplots()
    plt.plot(sorted_covariate_values, sorted_true_pi_r_values, label="True pi")
    plt.scatter(sorted_covariate_values, sorted_true_pi_r_values)
    plt.plot(sorted_covariate_values, sorted_pi_r_estimates, label="Estimated pi")
    plt.scatter(sorted_covariate_values, sorted_pi_r_estimates)
    plt.plot(sorted_covariate_values, sorted_yhat_pi_r, label="Predicted pi")
    plt.scatter(sorted_covariate_values, sorted_yhat_pi_r)
    plt.legend()
    plt.ylabel("pi_r: Fraction of resistant cells")
    plt.xlabel(cov_name)
    plt.title("pi_r: Fraction of resistant cells\nsorted by "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_pi_sorted_by_"+cov_name+".png")
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(sorted_covariate_values, sorted_true_g_r_values, label="True rho")
    plt.scatter(sorted_covariate_values, sorted_true_g_r_values)
    plt.plot(sorted_covariate_values, sorted_g_r_estimates, label="Estimated rho")
    plt.scatter(sorted_covariate_values, sorted_g_r_estimates)
    plt.plot(sorted_covariate_values, sorted_yhat_g_r, label="Predicted rho")
    plt.scatter(sorted_covariate_values, sorted_yhat_g_r)
    plt.legend()
    plt.ylabel("rho: growth rate of resistant cells")
    plt.xlabel(cov_name)
    #ax.set_yscale("log")
    plt.title("rho: growth rate of resistant cells\nsorted by "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_rho_sorted_by_"+cov_name+".png")
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(sorted_covariate_values, sorted_true_g_s_minus_K_values, label="True alpha-K")
    plt.scatter(sorted_covariate_values, sorted_true_g_s_minus_K_values)
    plt.plot(sorted_covariate_values, sorted_g_s_minus_K_estimates, label="Estimated alpha-K")
    plt.scatter(sorted_covariate_values, sorted_g_s_minus_K_estimates)
    plt.plot(sorted_covariate_values, sorted_yhat_g_s_minus_K, label="Predicted alpha-K")
    plt.scatter(sorted_covariate_values, sorted_yhat_g_s_minus_K)
    plt.legend()
    plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
    plt.xlabel(cov_name)
    plt.title("alpha-K: adjusted growth rate of sensitive cells\nsorted by "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/compare_true_and_estimated_alpha_minus_K_sorted_by_"+cov_name+".png")
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Visualization of model predictions on observed and test data:
    # Predictions for average values of other covariates, over the full true range of the indexed covariate in the current dataset
    N_testcases = 1000
    X_average_test_array = np.repeat([average_covariates], N_testcases, axis=0)
    print(X_average_test_array)

    test_covariate_values = np.linspace(min(covariate_values), max(covariate_values), N_testcases)
    print(test_covariate_values)
    X_average_test_array[:,covariate_index] = test_covariate_values
    print(X_average_test_array)

    true_test_pi_r = np.zeros(N_testcases)
    true_test_g_r = np.zeros(N_testcases)
    true_test_g_s_minus_K = np.zeros(N_testcases)
    for ii, X_covariates in enumerate(X_average_test_array):
        test_pi, test_g_r, test_g_s_minus_K = find_analytic_parameters(X_covariates)
        true_test_pi_r[ii] = test_pi
        true_test_g_r[ii] = test_g_r
        true_test_g_s_minus_K[ii] = test_g_s_minus_K
    
    # Make predictions 
    #model = LinearRegression()
    model = RandomForestRegressor(n_estimators=10)
    # ------------ pi_r -------------
    pi_r_model = model.fit(X, pi_r_estimates)
    yhat_test_pi_r = pi_r_model.predict(X_average_test_array)
    # ------------ g_r -------------
    g_r_model = model.fit(X[[choice in [1,3] for choice in model_choice]], g_r_estimates[[choice in [1,3] for choice in model_choice]])
    yhat_test_g_r = g_r_model.predict(X_average_test_array)
    # ------------ g_s_minus_K -------------
    g_s_minus_K_model = model.fit(X[[choice in [2,3] for choice in model_choice]], g_s_minus_K_estimates[[choice in [2,3] for choice in model_choice]])
    yhat_test_g_s_minus_K = g_s_minus_K_model.predict(X_average_test_array)
    
    print("yhat_test_pi_r", yhat_test_pi_r)
    print("max yhat_test_pi_r", max(yhat_test_pi_r))
    print("min(yhat_test_pi_r)", min(yhat_test_pi_r))

    print("yhat_test_g_r", yhat_test_g_r)
    print("max yhat_test_g_r", max(yhat_test_g_r))
    print("min(yhat_test_g_r)", min(yhat_test_g_r))

    print(yhat_test_g_s_minus_K)
    print("yhat_test_g_s_minus_K", yhat_test_g_s_minus_K)
    print("max yhat_test_g_s_minus_K", max(yhat_test_g_s_minus_K))
    print("min(yhat_test_g_s_minus_K)", min(yhat_test_g_s_minus_K))

    # Visualize predictions 
    fig, ax = plt.subplots()
    plt.plot(test_covariate_values, true_test_pi_r, label="True pi_r for average values in other variables")
    plt.scatter(test_covariate_values, true_test_pi_r)
    plt.plot(test_covariate_values, yhat_test_pi_r, label="Model prediction of pi_r for average values in other variables")
    plt.scatter(test_covariate_values, yhat_test_pi_r)
    plt.legend()
    plt.ylabel("pi_r: Fraction of resistant cells")
    plt.xlabel(cov_name)
    plt.title("Model predictions of pi_r (fraction of resistant cells)\nas a function of "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/model_predictions_of_pi_r_as_a_function_of_"+cov_name+".png")
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(test_covariate_values, true_test_g_r, label="True rho for average values in other variables")
    plt.scatter(test_covariate_values, true_test_g_r)
    plt.plot(test_covariate_values, yhat_test_g_r, label="Model prediction of rho for average values in other variables")
    plt.scatter(test_covariate_values, yhat_test_g_r)
    plt.legend()
    plt.ylabel("rho: growth rate of resistant cells")
    plt.xlabel(cov_name)
    #ax.set_yscale("log")
    plt.title("Model predictions of rho (growth rate of resistant cells)\nas a function of "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/model_predictions_of_rho_as_a_function_of_"+cov_name+".png")
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(test_covariate_values, true_test_g_s_minus_K, label="True alpha-K for average values in other variables")
    plt.scatter(test_covariate_values, true_test_g_s_minus_K)
    plt.plot(test_covariate_values, yhat_test_g_s_minus_K, label="Model prediction of alpha-K for average values in other variables")
    plt.scatter(test_covariate_values, yhat_test_g_s_minus_K)
    plt.legend()
    plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
    plt.xlabel(cov_name)
    plt.title("Model predictions of alpha-K (adjusted growth rate of sensitive cells)\nas a function of "+cov_name)
    plt.savefig("./plots/simulation_study_3_plots/model_predictions_of_alpha_minus_K_as_a_function_of_"+cov_name+".png")
    plt.tight_layout()
    plt.show()
    plt.close()
