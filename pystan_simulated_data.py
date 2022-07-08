# Purposes of this script: 
#   Take features, Mprotein observations and model choice of each patient (data)
#   Declare STAN model
#   Use STAN to sample parameter estimates (psi, theta)
#   This gives us a joint distribution of the map from features to drug response parameters
from utilities import *
import stan 

# Load period and patient definitions
picklefile = open('./binaries_and_pickles/patient_dictionary_simulation_study', 'rb')
patient_dictionary = pickle.load(picklefile)
picklefile.close()

# Load df_X_covariates
picklefile = open('./binaries_and_pickles/df_X_covariates_simulation_study', 'rb')
df_X_covariates = pickle.load(picklefile)
picklefile.close()

# Load model_choice
picklefile = open('./binaries_and_pickles/model_choice', 'rb')
model_choice = pickle.load(picklefile)
picklefile.close()

N = 102
P = 1 # One effect from HRD on one parameter
x = [[elem[1]] for elem in df_X_covariates]

y_pre_padding = [patient.Mprotein_values for name, patient in patient_dictionary.items()]
times_pre_padding = [patient.measurement_times for name, patient in patient_dictionary.items()]
len_y_each_patient = [len(elem) for elem in times_pre_padding]
max_len_y_each_patient = max(len_y_each_patient)

y = np.full((N, max_len_y_each_patient), np.inf)
times = np.full((N, max_len_y_each_patient), np.inf)
for i in range(N):
    for t in range(len_y_each_patient[i]):
        y[i,t] = y_pre_padding[i][t]
        times[i,t] = times_pre_padding[i][t]

# Assign the data to a dictionary
MM_data = {"N": N,
    "P": 1,
    "x": x,
    "y": y,
    "times": times,
    "max_len_y_each_patient": max_len_y_each_patient,
    "len_y_each_patient": len_y_each_patient,
    "model_choice": model_choice}

# Define the model
MM_model = """
data {
    int<lower=0> N; //Number of standalone treatment periods. If only one per patient, then equal to the number of patients. 
    int<lower=0> P;   // number of predictors
    int<lower=0> max_len_y_each_patient;   // Highest number of measurements for any patient 
    matrix[N, P] x;   // predictor matrix: HRD
    matrix[N, max_len_y_each_patient] y;      // outcome matrix where for the moment we assume every patient to have the same number of measurements 
    matrix[N, max_len_y_each_patient] times;      // measurement times 
    int len_y_each_patient[N]; // How many measurements there actually are for each patient 
    vector[N] model_choice; // 
}
parameters {
    // Covariate effect parameters
    vector[P] theta_Y_0; // effect of covariates on drug response parameters
    vector[P] theta_pi_r; // effect of covariates on drug response parameters
    vector[P] theta_rho; // effect of covariates on drug response parameters
    vector[P] theta_alpha; // effect of covariates on drug response parameters
    vector[P] theta_K; // effect of covariates on drug response parameters
    real<lower=0> sigma_theta_Y_0;
    real<lower=0> sigma_theta_pi_r;
    real<lower=0> sigma_theta_rho;
    real<lower=0> sigma_theta_alpha;
    real<lower=0> sigma_theta_K;
    // Case specific parameters 
    real<lower=0, upper=100> Y_0[N]; // 
    real<lower=0, upper=1> pi_r[N]; // 
    real<lower=0.001, upper=0.2> rho[N]; // 
    real<lower=0.001, upper=0.2> alpha[N]; // 
    real<lower=0.2, upper=1.0> K[N]; // 
    real<lower=0> mean_Y_0; // mean of group Y_0
    real<lower=0> sigma_Y_0; // standard deviation of group Y_0
    real<lower=0> mean_pi_r; // mean of group pi_r
    real<lower=0> sigma_pi_r; // standard deviation of group pi_r
    real<lower=0> mean_rho; // mean of group rho
    real<lower=0> sigma_rho; // standard deviation of group rho
    real<lower=0> mean_alpha; // mean of group alpha
    real<lower=0> sigma_alpha; // standard deviation of group alpha
    real<lower=0> mean_K; // mean of group K
    real<lower=0> sigma_K; // standard deviation of group K
    // Observation noise
    real<lower=0> sigma_obs; // observation error std dev
}
model {
    real mu; //declare the "single Y value predicted by parameters" variable
    vector[P] zero_vec;
    zero_vec = rep_vector(0, P);
    sigma_theta_rho ~ gamma(0.001, 0.0001);
    theta_Y_0 ~ normal(zero_vec, sigma_theta_Y_0); 
    theta_pi_r ~ normal(zero_vec, sigma_theta_pi_r); 
    theta_rho ~ normal(zero_vec, sigma_theta_rho); 
    theta_alpha ~ normal(zero_vec, sigma_theta_alpha); 
    theta_K ~ normal(zero_vec, sigma_theta_K); 
    for (i in 1:N){ // likelihood
        Y_0[i] ~ normal(mean_Y_0 - x*theta_Y_0, sigma_Y_0);
        pi_r[i] ~ normal(mean_pi_r - x*theta_pi_r, sigma_pi_r);
        rho[i] ~ normal(mean_rho - x*theta_rho, sigma_rho);
        alpha[i] ~ normal(mean_alpha - x*theta_alpha, sigma_alpha);
        K[i] ~ normal(mean_K - x*theta_K, sigma_K);
        for (t in 1:len_y_each_patient[i])
            if (model_choice[i] == 1) {
                mu = Y_0[i] * exp(times[i,t]*rho[i]);
                y[i,t] ~ normal(mu, sigma_obs);
            }
            else if (model_choice[i] == 2) {
                mu = Y_0[i] * exp(times[i,t]*(alpha[i]-K[i]));
                y[i,t] ~ normal(mu, sigma_obs);
            }
            else if (model_choice[i] == 3) {
                mu = Y_0[i] * (pi_r[i]* exp(times[i,t]*rho[i]) + (1-pi_r[i])*exp(times[i,t]*(alpha[i]-K[i])));
                y[i,t] ~ normal(mu, sigma_obs);
            }
    }
}
"""

# Get the posterior!
posterior = stan.build(MM_model, data=MM_data)
fit = posterior.sample(num_chains=4, num_samples=1000)
theta_pi_r = fit["theta_pi_r"]  # array with shape (8, 4000)
df = fit.to_frame()  # pandas `DataFrame, requires pandas
