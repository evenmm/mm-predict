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

def least_squares_objective_function(array_x, measurement_times, treatment_history, observations):
    Parameter_object_x = Parameters(Y_0=array_x[0], pi_r=array_x[1], g_r=array_x[2], g_s=array_x[3], k_1=0.000, sigma=global_sigma)
    predictions = measure_Mprotein_noiseless(Parameter_object_x, measurement_times, treatment_history)
    sumofsquares = np.sum((observations - predictions)**2)
    return sumofsquares

random_samples = np.random.uniform(0,1,len(ub))
x0_0 = lb + np.multiply(random_samples, (ub-lb))

lowest_f_value = np.inf
best_x = np.array([0,0,0,0])
for iteration in range(1000):
    random_samples = np.random.uniform(0,1,len(ub))
    x0 = lb + np.multiply(random_samples, (ub-lb))
    optimization_result = optimize.minimize(fun=least_squares_objective_function, x0=x0, args=(measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1), bounds=all_bounds, options={'disp':False})
    if optimization_result.fun < lowest_f_value:
        lowest_f_value = optimization_result.fun
        best_x = optimization_result.x

print("Compare truth with estimate:")
print("True x:", params_to_be_inferred)
print("Inferred x:", best_x)

f_value_at_truth = least_squares_objective_function(params_to_be_inferred, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)
x0_value = least_squares_objective_function(x0_0, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1)

print("f value at first x0:", x0_value)
print("f value at truth:", f_value_at_truth)
print("f value at estimate:", lowest_f_value)

estimated_parameters = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=0.000, sigma=global_sigma)
plot_true_mprotein_with_observations_and_treatments_and_estimate(parameters_patient_1, measurement_times_patient_1, treatment_history_patient_1, observed_M_protein_values_patient_1, estimated_parameters=estimated_parameters, PLOT_ESTIMATES=True)
"""