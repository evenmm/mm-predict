# Fit the linear regression to the training cases to learn the effect of the simulated binary variable HRD on each of the parameters pi, rho, alpha and K 
from utilities import *
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
warnings.simplefilter("ignore")
start_time = time.time()

N_patients_per_group = 34
N_patients = N_patients_per_group * 3

## Truth about patients (from model_selection_simulated_study.py): 
## Three patient groups: Hyperdiploid (A), middle guys (C) and Non-hyperdiploid (B)
## They have no history, and all patients get the same treatment for the same amount of days. The treatment is observed through all the interval.
## a) HRD completely resistant, Non-HRD completely sensitive, middel people have 0.1 resistant cells 
#average_pi_r_HRD = 1
#average_pi_r_middle_group = 0.1
#average_pi_r_non_HRD = 0
## b) Both groups partially resistant, but HRD more
##average_pi_r_HRD = 1
##average_pi_r_non_HRD = 0
#variance_in_pi_r_both_groups = 0
#observation_std_m_protein = 5 # sigma 
#
#Y_0_population = 50
#g_r_population = 0.002
#g_s_population = 0.010
#k_1_population = 0.015


# Load variables
picklefile = open('./binaries_and_pickles/model_choice', 'rb')
model_choice = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/df_X_covariates_simulation_study', 'rb')
df_X_covariates = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_parameters_simulation_study', 'rb')
Y_parameters = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_increase_or_not_simulation_study', 'rb')
Y_increase_or_not = pickle.load(picklefile) 
picklefile.close()

picklefile = open('./binaries_and_pickles/patient_dictionary_simulation_study', 'rb')
patient_dictionary = pickle.load(picklefile) 
picklefile.close()

# Get the parameters in array form 
pi_r_estimates = [param_object.pi_r for param_object in Y_parameters]
g_r_estimates = [param_object.g_r for param_object in Y_parameters]
g_s_estimates = [param_object.g_s for param_object in Y_parameters]
k_1_estimates = [param_object.k_1 for param_object in Y_parameters]

# Fit a linear regression for each 
X = df_X_covariates
y = pi_r_estimates
print(X)
print(y)
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
yhat = model.predict(X)
# summarize prediction
print(yhat)
print(model.coef_)
