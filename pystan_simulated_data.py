# Purposes of this script: 
#   Data: Mprotein observations y, times t, covariates x
#   Declare STAN model
#   Use STAN to sample parameter estimates: theta=function parameters (rho_s, rho_r, pi_r, phi), alpha and beta = effect of x parameters, eta = individual randomness 
#   This gives us a joint distribution of the map from features to y protein via drug response parameters
from utilities import *
import stan 

# Check population averages:
# true rho_s
picklefile = open('./binaries_and_pickles/Bayesian_true_rho_s_simulation_study', 'rb')
true_rho_s = pickle.load(picklefile)
picklefile.close()
# true rho_r
picklefile = open('./binaries_and_pickles/Bayesian_true_rho_r_simulation_study', 'rb')
true_rho_r = pickle.load(picklefile)
picklefile.close()
# true pi_r
picklefile = open('./binaries_and_pickles/Bayesian_true_pi_r_simulation_study', 'rb')
true_pi_r = pickle.load(picklefile)
picklefile.close()
# true psi_r
picklefile = open('./binaries_and_pickles/Bayesian_true_psi_simulation_study', 'rb')
true_psi = pickle.load(picklefile)
picklefile.close()

# These were specified in data generation, but the X variables will skew these upwards!!!! 
rho_s_population = -0.005
rho_r_population = 0.001
pi_r_population = 0.4
psi_population = 50
# Print population averages
print("Average true_rho_s:", np.mean(true_rho_s))
print("Average true_rho_r:", np.mean(true_rho_r))
print("Average true_pi_r:", np.mean(true_pi_r))
print("Average true_psi:", np.mean(true_psi))


# Load period and patient definitions
picklefile = open('./binaries_and_pickles/Bayesian_patient_dictionary_simulation_study', 'rb')
patient_dictionary = pickle.load(picklefile)
picklefile.close()

# Load df_X_covariates
picklefile = open('./binaries_and_pickles/Bayesian_df_X_covariates_simulation_study', 'rb')
df_X_covariates = pickle.load(picklefile)
picklefile.close()

N = 300 # num individuals 
P = 1 # One effect from HRD on one parameter, and one offset 
K = 3 # num parameters in theta not including psi.

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
M = len(y[0]) # num observations in y

# Data objects:
# matrix<lower=0>[N, M] y      # observations y
# matrix[N, M] t               # observation times t
# matrix<lower=0>[N, P] x      # covariates x
y = y 
t = times
x = [[elem[1]] for elem in df_X_covariates] # num covariates in x

# Assign the data to a dictionary
MM_data = {
    "N": N,
    "M": M,
    "P": P,
    "K": K,
    "y": y,
    "t": t,
    "x": x,
}

#for key, item in MM_data.items():
#    print(key)
#    print(item)

MM_model = """
// https://mc-stan.org/docs/2_19/stan-users-guide/multivariate-hierarchical-priors-section.html
functions {
    real predict_y(int N, int i, int j, vector rescaled_psi, vector rho_r, vector rho_s, vector pi_r, matrix t) {
        real y_hat; 
        y_hat = 50*rescaled_psi[i] * (pi_r[i]* exp(rho_r[i]*t[i][j]) + (1-pi_r[i])*exp(rho_s[i]*t[i][j]));
        return (y_hat);
    } 
}
data {
    int<lower=0> N;               // num individuals, 300
    int<lower=1> M;               // num observations in y, ca 10
    int<lower=1> K;               // num parameters in theta not including psi. Equal to 3.
    int<lower=1> P;               // num covariates in x
    matrix<lower=0>[N, M] y;               // observations y
    matrix<lower=0>[N, M] t;               // observation times t
    matrix[N, P] x;               // covariates x
}
parameters {
    //matrix[N, K] theta;           // parameters theta
    vector<upper=log(0.02)>[N] theta_1; // Theta for rho_s: Decay rate of sensitive cells 
    vector<upper=log(0.02)>[N] theta_2; // Theta for rho_r: Growth rate of resistant cells 
    vector[N] theta_3; // Theta for pi_r:  Fraction of resistant cells
    vector<lower=0>[N] rescaled_psi;       // 1/50 of True unobserved M protein at start of treatment  

    real<lower=0> sigma;         // observation error standard deviation
    vector<lower=0>[K] omega;  // variances for positive_rho_s, rho_r, pi_r

    vector<lower=0>[K] alpha;     // intercepts for positive_rho_s, rho_r, pi_r 
    matrix[K, P] beta; // coefficients beta for positive_rho_s, rho_r, pi_r
}
transformed parameters {
    // This follows Hilde's example. I assume that these are evaluated when they are needed in the model part 
    vector<lower=-0.2, upper=0>[N] rho_s;
    vector<lower=0, upper=0.2>[N] rho_r;
    vector<lower=0,upper=1>[N] pi_r;
    rho_s = - exp(theta_1);
    rho_r = exp(theta_2);
    pi_r = 1/(1+exp(-theta_3));

    matrix<lower=0>[N,M] y_hat; // Predicted y values 
    for (i in 1 : N) {
        for (j in 1 : M) {
            y_hat[i,j] = predict_y(N, i, j, rescaled_psi, rho_r, rho_s, pi_r, t);
        }
    }
}
model {
    // Priors
    // noise variance
    sigma ~ normal(0,1);
    for (l in 1:K)
        omega[l] ~ normal(0,1);

    // alpha and beta
    for (l in 1:K)
        alpha[l] ~ normal(0,1);
    for (l in 1:K) {
        for (b in 1 : P) {
            beta[l,b] ~ normal(0,1);
        }
    }

    // theta
    for (i in 1:N)
        theta_1[i] ~ normal(alpha[1] + beta[1] * x[i]', omega[1]);
    for (i in 1:N)
        theta_2[i] ~ normal(alpha[2] + beta[2] * x[i]', omega[2]);
    for (i in 1:N)
        theta_3[i] ~ normal(alpha[3] + beta[3] * x[i]', omega[3]);
    for (i in 1:N)
        rescaled_psi[i] ~ normal(0,1);

    // Likelihood
    for (i in 1 : N) {
        for (j in 1 : M) {
            y[i,j] ~ normal(y_hat[i,j],sigma);
        }
    }
}
generated quantities {
    vector<lower=0>[N] psi = 50*rescaled_psi;
}
"""

# Get the model!
MM_model = stan.build(MM_model, data=MM_data)
fit = MM_model.sample(num_chains=4, num_warmup=200, num_samples=1000)
#fit = MM_model.sample(num_chains=8, num_warmup=3000, num_samples=10000)

#MM_model = pystan.StanModel(file="./MM_model.stan")
#fit = MM_model.sampling(data=MM_data, iter=5000, chains=4, n_jobs=1)
#print(fit)

#medians <- summary(fit)$summary[ , "50%"]
print(fit)

print(fit["rho_s"].shape)
print(fit["rho_r"].shape)
print(fit["pi_r"].shape)
print(fit["psi"].shape)
#print(fit["sigma"].shape)
print(fit["pi_r"].shape)

print("Mean rho_s:", np.mean(fit["rho_s"]))
print("Mean rho_r:", np.mean(fit["rho_r"]))
print("Mean pi_r:", np.mean(fit["pi_r"]))
print("Mean psi:", np.mean(fit["psi"]))
#print("Mean sigma:", np.mean(fit["sigma"]))
print("Mean pi_r:", np.mean(fit["pi_r"]))

#print("pi_r:", fit["pi_r"])
print("Average of omega samples:", np.mean(fit["omega"], axis=1))

num_samples_here = len(fit["rho_s"][0])
num_patients_here = len(fit["rho_s"])
median_index = int(np.floor(num_samples_here/2))
medians_rho_s = [np.sort(fit["rho_s"][i])[median_index] for i in range(num_patients_here)]
medians_rho_r = [np.sort(fit["rho_r"][i])[median_index] for i in range(num_patients_here)]
medians_pi_r = [np.sort(fit["pi_r"][i])[median_index] for i in range(num_patients_here)]
medians_psi = [np.sort(fit["psi"][i])[median_index] for i in range(num_patients_here)]
medians_pi_r = [np.sort(fit["pi_r"][i])[median_index] for i in range(num_patients_here)]

print("\nMedians rho_s:",  medians_rho_s[:6])
print("Medians rho_r:",  medians_rho_r[:6])
print("Medians pi_r:",   medians_pi_r[:6])
print("Medians psi:",    medians_psi[:6])
print("Medians pi_r:",   medians_pi_r[:6])

print("\nMean rho_s, axis=1:", np.mean(fit["rho_s"], axis=1)[:6])
print("Mean rho_r, axis=1:", np.mean(fit["rho_r"], axis=1)[:6])
print("Mean pi_r, axis=1:", np.mean(fit["pi_r"], axis=1)[:6])
print("Mean psi, axis=1:", np.mean(fit["psi"], axis=1)[:6])
print("Mean pi_r, axis=1:", np.mean(fit["pi_r"], axis=1)[:6])
#print("\nMean sigma, axis=1:", np.mean(fit["sigma"], axis=1)[:6])


"""

print(fit["y_hat"].shape)
print("y_hat:", np.mean(fit["y_hat"]))
print("y_hat, axis=1:", np.mean(fit["y_hat"], axis=1))
print("y_hat, axis=1:", np.mean(fit["y_hat"], axis=0))
print("y_hat, axis=1:", np.mean(fit["y_hat"], axis=2))

picklefile = open('./binaries_and_pickles/stan_fit', 'wb')
pickle.dump(np.array(fit), picklefile)
picklefile.close()

#pi_r = fit["pi_r"]  # array with shape (8, 4000)
#df = fit.to_frame()  # pandas `DataFrame, requires pandas


def plot_trace(param, param_name='parameter'):
   #Plot the trace and posterior of a parameter.
  
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

#plot_trace(fit["pi_r"], param_name="pi_r")

##permuted_results = fit.extract(permuted=True)
##Lmu = permuted_results['pi_r']
#Lmu = fit['pi_r']
#meanMu = np.mean(Lmu)
#plt.figure()
#plt.title("Posterior Distribution of pi_r     Mean = {0:.7}".format(meanMu))
#plt.hist(Lmu, bins = 50)
#plt.show()
"""
