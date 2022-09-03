# Fit the linear regression to the training cases to learn the effect of the simulated binary variable HRD on each of the parameters pi, rho, alpha and K 
from utilities import *
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FormatStrFormatter
warnings.simplefilter("ignore")
start_time = time.time()

# Load saved values
picklefile = open('./binaries_and_pickles/df_X_covariates_simulation_study_version_2', 'rb')
df_X_covariates = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_parameters_simulation_study_version_2', 'rb')
Y_parameters = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_increase_or_not_simulation_study_version_2', 'rb')
Y_increase_or_not = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/patient_dictionary_simulation_study_version_2', 'rb')
patient_dictionary = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/model_choice_version_2', 'rb')
model_choice = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/chosen_estimates_version_2', 'rb')
chosen_estimates = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/true_parameters_version_2', 'rb')
true_parameters = pickle.load(picklefile)
picklefile.close()

N_patients = len(model_choice)

# Get the parameters in array form 
pi_r_estimates = [param_object.pi_r for param_object in chosen_estimates]
g_r_estimates = [param_object.g_r for param_object in chosen_estimates]
g_s_estimates = [param_object.g_s for param_object in chosen_estimates]
k_1_estimates = [param_object.k_1 for param_object in chosen_estimates]
g_s_minus_K_estimates = [g_s_estimates[ii] - k_1_estimates[ii] for ii in range(len(k_1_estimates))]

true_pi_r_values = [param_object.pi_r for param_object in true_parameters]
true_g_r_values = [param_object.g_r for param_object in true_parameters]
true_g_s_values = [param_object.g_s for param_object in true_parameters]
true_k_1_values = [param_object.k_1 for param_object in true_parameters]
true_g_s_minus_K_values = [true_g_s_values[ii] - true_k_1_values[ii] for ii in range(len(true_k_1_values))]

# Compare truth with estimate
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.plot(true_pi_r_values, label="True pi")
plt.plot(pi_r_estimates, label="Estimated pi")
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Days since treatment start")
plt.xticks(np.linspace(0,N_patients,6), 100*np.linspace(0,N_patients,6))
plt.savefig("./plots/simulation_study_2_plots/compare_true_and_estimated_pi.png")
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.plot(true_g_r_values, label="True rho")
plt.plot(g_r_estimates, label="Estimated rho")
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Days since treatment start")
plt.xticks(np.linspace(0,N_patients,6), 100*np.linspace(0,N_patients,6))
plt.savefig("./plots/simulation_study_2_plots/compare_true_and_estimated_rho.png")
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.plot(true_g_s_minus_K_values, label="True alpha-K")
plt.plot(g_s_minus_K_estimates, label="Estimated alpha-K")
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Days since treatment start")
plt.xticks(np.linspace(0,N_patients,6), 100*np.linspace(0,N_patients,6))
plt.savefig("./plots/simulation_study_2_plots/compare_true_and_estimated_alpha_minus_K.png")
plt.show()

# Fit a linear regression for each identifiable parameter
# ------------ pi_r -------------
X = df_X_covariates
y = pi_r_estimates
print(X)
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
X = df_X_covariates
y = g_r_estimates
print(X)
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
X = df_X_covariates
y = g_s_minus_K_estimates
print(X)
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
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.plot(true_pi_r_values, label="True pi")
plt.plot(pi_r_estimates, label="Estimated pi")
plt.plot(yhat_pi_r, label="Machine learning prediction")
plt.legend()
plt.ylabel("pi_r: Fraction of resistant cells")
plt.xlabel("Days since treatment start")
plt.xticks(np.linspace(0,N_patients,6), 100*np.linspace(0,N_patients,6))
plt.savefig("./plots/simulation_study_2_plots/compare_true_and_estimated_pi.png")
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.plot(true_g_r_values, label="True rho")
plt.plot(g_r_estimates, label="Estimated rho")
plt.plot(yhat_pi_r, label="Machine learning prediction")
plt.legend()
plt.ylabel("rho: growth rate of resistant cells")
plt.xlabel("Days since treatment start")
plt.xticks(np.linspace(0,N_patients,6), 100*np.linspace(0,N_patients,6))
plt.savefig("./plots/simulation_study_2_plots/compare_true_and_estimated_rho.png")
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.plot(true_g_s_minus_K_values, label="True alpha-K")
plt.plot(g_s_minus_K_estimates, label="Estimated alpha-K")
plt.plot(yhat_pi_r, label="Machine learning prediction")
plt.legend()
plt.ylabel("alpha-K: adjusted growth rate of sensitive cells")
plt.xlabel("Days since treatment start")
plt.xticks(np.linspace(0,N_patients,6), 100*np.linspace(0,N_patients,6))
plt.savefig("./plots/simulation_study_2_plots/compare_true_and_estimated_alpha_minus_K.png")
plt.show()

