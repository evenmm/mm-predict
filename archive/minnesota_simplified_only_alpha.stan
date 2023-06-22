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
    //vector<upper=log(0.02)>[N] theta_1; // Theta for rho_s: Decay rate of sensitive cells 
    //vector<upper=log(0.02)>[N] theta_2; // Theta for rho_r: Growth rate of resistant cells 
    //vector[N] theta_3; // Theta for pi_r:  Fraction of resistant cells
    vector<lower=0>[N] rescaled_psi;       // 1/50 of True unobserved M protein at start of treatment  

    real<lower=0> sigma;         // observation error standard deviation
    vector<lower=0>[K] omega;  // variances for positive_rho_s, rho_r, pi_r

    vector<lower=0>[K] alpha;     // intercepts for positive_rho_s, rho_r, pi_r 
    //matrix[K, P] beta; // coefficients beta for positive_rho_s, rho_r, pi_r
}
transformed parameters {
    // This follows Hilde's example. I assume that these are evaluated when they are needed in the model part 
    vector[N] theta_1; // Theta for rho_s: Decay rate of sensitive cells 
    vector[N] theta_2; // Theta for rho_r: Growth rate of resistant cells 
    vector[N] theta_3; // Theta for pi_r:  Fraction of resistant cells
    for (i in 1:N)
        theta_1[i] = log(0.005) + [1] * x[i]'; //Desired alpha[1] = log(0.005)
    for (i in 1:N)
        theta_2[i] = log(0.001) + [1] * x[i]'; //Desired alpha[2] = log(0.001)
    for (i in 1:N)
        theta_3[i] = alpha[3] + [1] * x[i]'; //Desired alpha[3] = logit(0.4) = -0.0969

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
    sigma ~ normal(0,0.1);
    for (l in 1:K)
        omega[l] ~ normal(0,0.1);

    // alpha and beta
    for (l in 1:K)
        alpha[l] ~ normal(0,0.1);
    //for (l in 1:K) {
    //    for (b in 1 : P) {
    //        beta[l,b] ~ normal(0,1);
    //    }
    //}

    //// theta
    //for (i in 1:N)
    //    theta_1[i] ~ normal(alpha[1] + beta[1] * x[i]', omega[1]);
    //for (i in 1:N)
    //    theta_2[i] ~ normal(alpha[2] + beta[2] * x[i]', omega[2]);
    //for (i in 1:N)
    //    theta_3[i] ~ normal(alpha[3] + beta[3] * x[i]', omega[3]);
    for (i in 1:N)
        rescaled_psi[i] ~ normal(0,0.1);

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
