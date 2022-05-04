from utilities import *

#####################################
# Generate data 
#####################################
# Shared parameters
#####################################
# Drug effects: 
# map id=1 to effect=k_1
# map id=2 to effect=k_2
# ...

## Patient specific parameters
# No drug:   A doubling time of 2 months = 60 days means that X1/X0 = 2   = e^(60*alpha) ==> alpha = log(2)/60   = approx.  0.005 1/days
# With drug: A halving  time of 2 months = 60 days means that X1/X0 = 1/2 = e^(60*alpha) ==> alpha = log(1/2)/60 = approx. -0.005 1/days
# Encoding cost of resistance giving resistant cells a lower base growth rate than sensitive cells 

"""
#####################################
# Patient 1
#####################################
# With drug effect and growth rate parameters:
#parameters_patient_1 = Parameters(Y_0=50, pi_r=0.10, g_r=0.008, g_s=0.010, k_1=0.020, sigma=global_sigma)
# With bulk growth rates on treatment:
#parameters_patient_1 = Parameters(Y_0=50, pi_r=0.01, g_r=0.008, g_s=-0.010, k_1=0.000, sigma=global_sigma)
parameters_patient_1 = Parameters(Y_0=50, pi_r=0.01, g_r=0.040, g_s=-0.200, k_1=0.000, sigma=global_sigma)

# Measure M protein
Mprotein_recording_interval_patient_1 = 10 #every X days
N_Mprotein_measurements_patient_1 = 5 # for N*X days
measurement_times_patient_1 = Mprotein_recording_interval_patient_1 * np.linspace(0,N_Mprotein_measurements_patient_1,N_Mprotein_measurements_patient_1+1)

# Simplest case: Only one treatment period
T_1_p_1 = Treatment(start=0, end=measurement_times_patient_1[-1], id=1)
treatment_history_patient_1 = [T_1_p_1]

patient_1 = Patient(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
patient_1.plot()

print("Measurement times patient 1", measurement_times_patient_1)

noiseless_M_protein_values_patient_1 = measure_Mprotein_naive(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
print("noiseless_M_protein_values_patient_1, Naive method:\n", noiseless_M_protein_values_patient_1)

noiseless_M_protein_values_patient_1 = measure_Mprotein_noiseless(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
print("noiseless_M_protein_values_patient_1, efficient method:\n", noiseless_M_protein_values_patient_1)

observed_M_protein_values_patient_1 = measure_Mprotein_with_noise(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1)
print("observed_M_protein_values_patient_1, efficient method:\n", observed_M_protein_values_patient_1)

#####################################
# Plot data 
#####################################
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)

#####################################
# Inference
#####################################
# For inferring both growth rates, k_D and sigma:
##      Y_0, pi_r,   g_r,   g_s,   k_1=0.020
#lb = [   0,    0, -0.10, -0.10,  0.00, 1e-6]
#ub = [1000,    1,  0.10,  0.10,  0.50,  5e4]
params_to_be_inferred = np.array([parameters_patient_1.Y_0, parameters_patient_1.pi_r, parameters_patient_1.g_r, parameters_patient_1.g_s])
# For inferring bulk growth rates
#               Y_0, pi_r,   g_r,   g_s,
lb = np.array([   0,    0, -1.00, -2.00])
ub = np.array([1000,    1,  1.00,  2.00])
bounds_Y_0 = (lb[0], ub[0])
bounds_pi_r = (lb[1], ub[1])
bounds_g_r = (lb[2], ub[2])
bounds_g_s = (lb[3], ub[3])
#bounds_Y_0 =     (0, 1000)
#bounds_pi_r =    (0,    1)
#bounds_g_r = (-0.10, 0.10)
#bounds_g_s = (-0.10, 0.10)

all_bounds = (bounds_Y_0, bounds_pi_r, bounds_g_r, bounds_g_s)

random_samples = np.random.uniform(0,1,len(ub))
x0_0 = lb + np.multiply(random_samples, (ub-lb))

lowest_f_value = np.inf
best_x = np.array([0,0,0,0])
for iteration in range(1000):
    random_samples = np.random.uniform(0,1,len(ub))
    x0 = lb + np.multiply(random_samples, (ub-lb))
    optimization_result = optimize.minimize(fun=least_squares_objective_function_patient_1, x0=x0, args=(measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1), bounds=all_bounds, options={'disp':False})
    if optimization_result.fun < lowest_f_value:
        lowest_f_value = optimization_result.fun
        best_x = optimization_result.x

print("Compare truth with estimate:")
print("True x:", params_to_be_inferred)
print("Inferred x:", best_x)

f_value_at_truth = least_squares_objective_function_patient_1(params_to_be_inferred, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)
x0_value = least_squares_objective_function_patient_1(x0_0, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)

print("f value at first x0:", x0_value)
print("f value at truth:", f_value_at_truth)
print("f value at estimate:", lowest_f_value)

estimated_parameters_patient_1 = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=0.000, sigma=global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1, estimated_parameters=estimated_parameters_patient_1, PLOT_ESTIMATES=True)
"""

#####################################
# Patient 2
#####################################
# 1) Simulate data
# With k drug effect and growth rate parameters: 
parameters_patient_2 = Parameters(Y_0=50, pi_r=0.10, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)
#parameters_patient_2 = Parameters(Y_0=50, pi_r=0.10, g_r=2e-3, g_s=1e-2, k_1=3e-2, sigma=global_sigma)

# Measure M protein
Mprotein_recording_interval_patient_2 = 5 #every X days
#Mprotein_recording_interval_patient_2 = 200 #every X days
N_Mprotein_measurements_patient_2 = 8 # for N*X days
measurement_times_patient_2 = Mprotein_recording_interval_patient_2 * np.linspace(0,N_Mprotein_measurements_patient_2,N_Mprotein_measurements_patient_2+1)

# Define history
treatment_history_patient_2 = [
    Treatment(start=0, end=measurement_times_patient_2[4], id=1),
    Treatment(start=measurement_times_patient_2[4], end=measurement_times_patient_2[8], id=0),
    ]

patient_2 = Patient(parameters_patient_2, measurement_times_patient_2, treatment_history_patient_2, covariates = [0,1])
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_2, patient_2, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 2")

## Inference
# For inferring both k_1 and growth rates
#               Y_0, pi_r,   g_r,   g_s,  k_1
lb = np.array([   0,    0,  0.00,  0.00, 0.00])
ub = np.array([1000,    1,  2.00,  2.00, 2.00])

param_array_patient_2 = parameters_patient_2.to_array_without_sigma()

random_samples = np.random.uniform(0,1,len(ub))
x0_0 = lb + np.multiply(random_samples, (ub-lb))

best_optim = infer_parameters_simulated_patient(patient_2, lb, ub)
best_x = best_optim.x 
lowest_f_value = best_optim.fun

print("Compare truth with estimate:")
print("True x:", param_array_patient_2)
print("Inferred x:", best_x)

f_value_at_truth = least_squares_objective_function(param_array_patient_2, patient_2)
x0_value = least_squares_objective_function(x0_0, patient_2)

print("f value at first x0:", x0_value)
print("f value at truth:", f_value_at_truth)
print("f value at estimate:", lowest_f_value)

estimated_parameters_patient_2 = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_2, patient_2, estimated_parameters=estimated_parameters_patient_2, PLOT_ESTIMATES=True, plot_title="Patient 2")

#####################################################
# Learn effect of history on drug response parameters 
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# X is a matrix containing the covariates of the patients 
# y is a matrix containing the estimates of their parameters under the drug of interest
# To evaluate the learning, compare predicted parameters with ESTIMATED parameters of a patient with similar covariates
# To evaluate the whole method (prediction based on estimated parameters), compare the prediction with TRUE parameters of a patient with similar covariates
# Covariates: 
# First covariate: If 0 then pi_r = 0.10. If 1 then pi_r = 0.90. 
# Second covariate: 1 if treatment is given before break. Irrelevant wrt parameters.

# create dataset
Mprotein_recording_interval = 5 #every X days
N_Mprotein_measurements = 8 # for N*X days
measurement_times = Mprotein_recording_interval * np.linspace(0,N_Mprotein_measurements,N_Mprotein_measurements+1)

def get_parameters_from_optimresult(best_x, sigma):
    return Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=sigma)

# Patient 3
parameters_patient_3 = Parameters(Y_0=10, pi_r=0.10, g_r=0.030, g_s=0.100, k_1=0.400, sigma=global_sigma)
treatment_history_patient_3 = [
    Treatment(start=0, end=measurement_times[4], id=0),
    Treatment(start=measurement_times[4], end=measurement_times[8], id=1),
    ]
patient_3 = Patient(parameters_patient_3, measurement_times, treatment_history_patient_3, covariates = [0,0])
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_3, patient_3, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 3")
best_optim_3 = infer_parameters_simulated_patient(patient_3, lb, ub)
best_x_3 = best_optim_3.x 
lowest_f_value_3 = best_optim_3.fun
estimated_parameters_patient_3 = get_parameters_from_optimresult(best_x_3, global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_3, patient_3, estimated_parameters=estimated_parameters_patient_3, PLOT_ESTIMATES=True, plot_title="Patient 3")
# Patient 4
parameters_patient_4 = Parameters(Y_0=50, pi_r=0.90, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)
treatment_history_patient_4 = [
    Treatment(start=0, end=measurement_times[4], id=1),
    Treatment(start=measurement_times[4], end=measurement_times[8], id=0),
    ]
patient_4 = Patient(parameters_patient_4, measurement_times, treatment_history_patient_4, covariates = [1,0])
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_4, patient_4, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 4")
best_optim_4 = infer_parameters_simulated_patient(patient_4, lb, ub)
best_x_4 = best_optim_4.x 
lowest_f_value_4 = best_optim_4.fun
estimated_parameters_patient_4 = get_parameters_from_optimresult(best_x_4, global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_4, patient_4, estimated_parameters=estimated_parameters_patient_4, PLOT_ESTIMATES=True, plot_title="Patient 4")
# Patient 5
parameters_patient_5 = Parameters(Y_0=50, pi_r=0.90, g_r=0.020, g_s=0.100, k_1=0.300, sigma=global_sigma)
treatment_history_patient_5 = [
    Treatment(start=0, end=measurement_times[4], id=0),
    Treatment(start=measurement_times[4], end=measurement_times[8], id=1),
    ]
patient_5 = Patient(parameters_patient_5, measurement_times, treatment_history_patient_5, covariates = [1,1])
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_5, patient_5, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 5")
best_optim_5 = infer_parameters_simulated_patient(patient_5, lb, ub)
best_x_5 = best_optim_5.x 
lowest_f_value_5 = best_optim_5.fun
estimated_parameters_patient_5 = get_parameters_from_optimresult(best_x_5, global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_5, patient_5, estimated_parameters=estimated_parameters_patient_5, PLOT_ESTIMATES=True, plot_title="Patient 5")

#patient_3 = deepcopy(patient_2)
#patient_4 = deepcopy(patient_2)
#patient_4.covariates = [1,0]
#patient_5 = deepcopy(patient_4)

# n = 4, p = 1, dim(Y) = 4
# X is a matrix containing the covariates of the patients 
X = np.zeros((4,2))
X[0,:] = patient_2.covariates
X[1,:] = patient_3.covariates
X[2,:] = patient_4.covariates
X[3,:] = patient_5.covariates
print(X)
# true_y is a matrix containing true parameters under the drug of interest
true_y = np.zeros((4,4))
true_y[0,:] = parameters_patient_2.to_array_without_sigma()[1:5]
true_y[1,:] = parameters_patient_3.to_array_without_sigma()[1:5]
true_y[2,:] = parameters_patient_4.to_array_without_sigma()[1:5]
true_y[3,:] = parameters_patient_5.to_array_without_sigma()[1:5]
print(true_y)
# y is a matrix containing estimates of the patients' parameters under the drug of interest
y = np.zeros((4,4))
y[0,:] = estimated_parameters_patient_2.to_array_without_sigma()[1:5]
y[1,:] = estimated_parameters_patient_3.to_array_without_sigma()[1:5]
y[2,:] = estimated_parameters_patient_4.to_array_without_sigma()[1:5]
y[3,:] = estimated_parameters_patient_5.to_array_without_sigma()[1:5]
print(y)
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# predict on test patient
test_covariates = [0,1]
print(test_covariates)
#row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([test_covariates])
# summarize prediction
print(yhat[0])

