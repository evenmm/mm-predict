# Purposes of this script: 
#   Data: Covariates X, Mprotein observations y and model choice for each patient
#   Declare STAN model
#   Use STAN to sample parameter estimates (psi, theta, mean_psi, sigma)
#   This gives us a joint distribution of the map from features to drug response parameters
from utilities import *
#import pystan 
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
P = 1 # One effect from HRD on one parameter, and one offset 
x = [[elem[1]] for elem in df_X_covariates]

y_pre_padding = [patient.Mprotein_values for _, patient in patient_dictionary.items()]
times_pre_padding = [patient.measurement_times for _, patient in patient_dictionary.items()]
len_y_each_patient = [len(elem) for elem in times_pre_padding]
max_len_y = max(len_y_each_patient)

y = [[np.nan for tt in range(max_len_y)] for ii in range(N)]
times = [[np.nan for tt in range(max_len_y)] for ii in range(N)]
#y = np.full((N, max_len_y), np.inf)
#times = np.full((N, max_len_y), np.inf)
for i in range(N):
    for t in range(len_y_each_patient[i]):
        y[i][t] = y_pre_padding[i][t]
        times[i][t] = times_pre_padding[i][t]

# Assign the data to a dictionary
MM_data = {
    "N": N,
    "P": 1,
    "x": x,
    "max_len_y": max_len_y,
    "len_y_each_patient": len_y_each_patient,
    "times": times,
    "y": y,
    "model_choice": model_choice}

#for key, item in MM_data.items():
#    print(key)
#    print(item)

MM_model = """
functions {
    // real loglikelihood_of_individual_psi_parameters(int N, int i, int len_y_i, vector[N] model_choice, vector[N] Y_0, vector[N] pi_r, vector[N] rho, vector[N] alpha, vector[N] K, vector[len_y_i] observations_i, real sigma_y) {
    //     sum_of_squares = 0 
    //     for (t in 1:len_y_i)
    //         if (model_choice[i] == 1) {
    //             mu = Y_0[i] * exp(times[i,t]*rho[i]);
    //         }
    //         else if (model_choice[i] == 2) {
    //             mu = Y_0[i] * exp(times[i,t]*(alpha[i]-K[i]));
    //         }
    //         else if (model_choice[i] == 3) {
    //             mu = Y_0[i] * (pi_r[i]* exp(times[i,t]*rho[i]) + (1-pi_r[i])*exp(times[i,t]*(alpha[i]-K[i])));
    //         }
    //         sum_of_squares = sum_of_squares + (mu - observations_i[t])**2
    //     loglikelihood = - (len_y_i/2)*np.log(2*np.pi*sigma_y**2) - sumofsquares/(2*sigma_y**2)
    //     return (loglikelihood);
    // } 
    real model_prediction(int N, int i, int t_index, array[] int model_choice, array[] real Y_0, array[] real pi_r, array[] real rho, array[] real alpha_minus_K, array[,] real times) {
        real mu; 
        if (model_choice[i] == 1) {
            mu = Y_0[i] * exp(times[i][t_index]*rho[i]);
        }
        else if (model_choice[i] == 2) {
            mu = Y_0[i] * exp(times[i][t_index]*(alpha_minus_K[i])); //alpha[i]-K[i]));
        }
        else { //if (model_choice[i] == 3) {
            mu = Y_0[i] * (pi_r[i]* exp(times[i][t_index]*rho[i]) + (1-pi_r[i])*exp(times[i][t_index]*(alpha_minus_K[i])));
        }
        return (mu);
    } 
}

data {
    int N; //Number of standalone treatment periods. If only one per patient, then equal to the number of patients. 
    int P;   // number of covariates
    array[N] vector[P] x;   // covariate matrix
    int max_len_y;   // Highest number of measurements for any patient 
    array[N] int len_y_each_patient; // How many measurements there actually are for each patient 
    array[N,max_len_y] real times;      // measurement times 
    array[N,max_len_y] real y;      // outcome matrix where for the moment we assume every patient to have the same number of measurements 
    array[N] int model_choice; // 
}

parameters {
    // Covariate effects 
    vector[P] theta_pi_r; // effect of covariates on drug response parameters
    vector[P] theta_rho; // effect of covariates on drug response parameters
    // vector[P] theta_alpha; // effect of covariates on drug response parameters
    // vector[P] theta_K; // effect of covariates on drug response parameters
    vector[P] theta_alpha_minus_K; // effect of covariates on drug response parameters

    // Intercepts: 
    real<lower=0> mean_Y_0; // mean of group Y_0
    real<lower=0, upper=1> mean_pi_r; // mean of group pi_r
    real mean_rho; // mean of group rho
    // real<lower=0> mean_alpha; // mean of group alpha
    // real<lower=0> mean_K; // mean of group K
    real mean_alpha_minus_K; // mean of group K

    // Parameters: 
    array[N] real<lower=0> Y_0; // real<lower=0> Y_0[N]; // 
    array[N] real<lower=0, upper=1> pi_r; // real<lower=0, upper=1>
    array[N] real<lower=0.001, upper=0.2> rho; // real<lower=0.001, upper=0.2>
    // real<lower=0.001, upper=0.2> alpha[N]; // 
    // real<lower=0.2, upper=1.0> K[N]; // 
    array[N] real<lower=-0.999, upper=0> alpha_minus_K; // real<lower=-0.999, upper=0>

    // Observation noise
    real<lower=0, upper=10000> sigma_obs; // observation error std dev
}

//transformed parameters {
//    // Model predictions of M protein 
//    array[N,max_len_y] real pred_M_protein;
//    profile("likelihood-model_predictions_M_protein") {
//    for (i in 1:N){
//        for (t_index in 1:len_y_each_patient[i]) {
//            pred_M_protein[i][t_index] = model_prediction(N, i, t_index, model_choice, Y_0, pi_r, rho, alpha_minus_K, times);
//        }
//    }
//    }
//}

model {
    // Priors for covariate effects: Assume all centered about 0.1 to see if we escape from there. Then center around 0 (Ridge penalty)
    //int p;
    for (p in 1:P) {
        theta_pi_r[p] ~ normal(0.1, 1);
        theta_rho[p] ~ normal(0.1, 1);
        theta_alpha_minus_K[p] ~ normal(0.1, 1);
        // theta_alpha[p] ~ normal(0.1, 1); 
        // theta_K[p] ~ normal(0.1, 1); 
    }
    // Priors for intercepts
    mean_Y_0 ~ normal(52, 20);
    mean_pi_r ~ normal(0.5, 0.3);
    mean_rho ~ normal(0.005, 0.005);
    mean_alpha_minus_K ~ normal(-0.005, 0.005);

    // Priors for parameters : based on covariate effects 
    for (i in 1:N) {
        Y_0[i] ~ normal(mean_Y_0, 20); 
        pi_r[i] ~ normal(mean_pi_r - dot_product(x[i], theta_pi_r), 0.1);
        rho[i] ~ normal(mean_rho - dot_product(x[i], theta_rho), 0.1);
        // alpha[i] ~ normal(mean_alpha - dot_product(x, theta_alpha), 0.1);
        // K[i] ~ normal(mean_K - dot_product(x, theta_K), 0.1);
        alpha_minus_K[i] ~ normal(mean_alpha_minus_K - dot_product(x[i], theta_alpha_minus_K), 0.1);
    }
    // Likelihood for Mprotein observations:
    for (i in 1:N){ // likelihood
        for (t_index in 1:len_y_each_patient[i]) {
            y[i][t_index] ~ normal(model_prediction(N, i, t_index, model_choice, Y_0, pi_r, rho, alpha_minus_K, times), sigma_obs);
            //y[i][t_index] ~ normal(pred_M_protein[i][t_index], sigma_obs);
        }
    }
}

"""

# Get the model!
MM_model = stan.build(MM_model, data=MM_data)
fit = MM_model.sample(num_chains=8, num_samples=100000)
#fit = MM_model.sample(num_chains=12, iter=5000, num_samples=1000)

#MM_model = pystan.StanModel(file="./MM_model.stan")
#fit = MM_model.sampling(data=MM_data, iter=5000, chains=4, n_jobs=1)
print(fit)

print(fit["theta_pi_r"].shape)
print(fit["theta_rho"].shape)
print(fit["theta_alpha_minus_K"].shape)
print(fit["mean_Y_0"].shape)
print(fit["mean_pi_r"].shape)
print(fit["mean_rho"].shape)
print(fit["mean_alpha_minus_K"].shape)
print(fit["Y_0"].shape)
print(fit["pi_r"].shape)
print(fit["rho"].shape)
print(fit["alpha_minus_K"].shape)
print(fit["sigma_obs"].shape)

print("theta_pi_r:", np.mean(fit["theta_pi_r"]))
print("theta_rho:", np.mean(fit["theta_rho"]))
print("theta_alpha_minus_K:", np.mean(fit["theta_alpha_minus_K"]))
print("mean_Y_0:", np.mean(fit["mean_Y_0"]))
print("mean_pi_r:", np.mean(fit["mean_pi_r"]))
print("mean_rho:", np.mean(fit["mean_rho"]))
print("mean_alpha_minus_K:", np.mean(fit["mean_alpha_minus_K"]))
print("Y_0:", np.mean(fit["Y_0"]))
print("rho:", np.mean(fit["rho"]))
print("alpha_minus_K:", np.mean(fit["alpha_minus_K"]))
print("sigma_obs:", np.mean(fit["sigma_obs"]))
print("pi_r:", np.mean(fit["pi_r"]))
print("pi_r, axis=1:", np.mean(fit["pi_r"], axis=1))


picklefile = open('./binaries_and_pickles/stan_fit', 'wb')
pickle.dump(np.array(fit), picklefile)
picklefile.close()

#theta_pi_r = fit["theta_pi_r"]  # array with shape (8, 4000)
#df = fit.to_frame()  # pandas `DataFrame, requires pandas


def plot_trace(param, param_name='parameter'):
  """Plot the trace and posterior of a parameter."""
  
  # Summary statistics
  mean = np.mean(param)
  median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
  
  # Plotting
  plt.subplot(2,1,1)
  plt.plot(param)
  plt.xlabel('samples')
  plt.ylabel(param_name)
  plt.axhline(mean, color='r', lw=2, linestyle='--')
  plt.axhline(median, color='c', lw=2, linestyle='--')
  plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  plt.title('Trace and Posterior Distribution for {}'.format(param_name))

  plt.subplot(2,1,2)
  plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
  plt.xlabel(param_name)
  plt.ylabel('density')
  plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
  plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  
  plt.gcf().tight_layout()
  plt.legend()

#plot_trace(fit["theta_pi_r"], param_name="theta_pi_r")

"""
#permuted_results = fit.extract(permuted=True)
#Lmu = permuted_results['theta_pi_r']
Lmu = fit['theta_pi_r']
meanMu = np.mean(Lmu)
plt.figure()
plt.title("Posterior Distribution of theta_pi_r     Mean = {0:.7}".format(meanMu))
plt.hist(Lmu, bins = 50)
plt.show()
"""
