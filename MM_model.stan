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
