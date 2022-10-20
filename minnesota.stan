// https://mc-stan.org/docs/2_19/stan-users-guide/multivariate-hierarchical-priors-section.html
functions {
    //real predict_y(int N, int i, int j, array[] real psi, array[] real rho_r, array[] real positive_rho_s, array[] real pi_r, array[] real t) {
    real predict_y(int N, int i, int j, vector psi, vector rho_r, vector positive_rho_s, vector pi_r, matrix t) {
        real y_hat; 
        y_hat = psi[i] * (pi_r[i]* exp(rho_r[i]*t[i][j]) + (1-pi_r[i])*exp(-positive_rho_s[i]*t[i][j]));
        return (y_hat);
    } 
}
data {
    int<lower=0> N;               // num individuals, 300
    int<lower=1> M;               // num observations in y, ca 10
    int<lower=1> K;               // num parameters in theta not including psi. Equal to 3.
    int<lower=1> P;               // num covariates in x
    // all these should be <lower=0>
    matrix<lower=0>[N, M] y;               // observations y
    matrix<lower=0>[N, M] t;               // observation times t
    matrix<lower=0>[N, P] x;               // covariates x
}
parameters {
    //matrix[N, K] theta;           // parameters theta
    vector<lower=0, upper=0.2>[N] positive_rho_s;     // Decay rate of sensitive cells 
    vector<lower=0, upper=0.2>[N] rho_r;     // Growth rate of resistant cells 
    vector<lower=0, upper=1>[N] pi_r;      // Fraction of resistant cells
    vector<lower=0>[N] psi;       // True unobserved M protein at start of treatment  

    real<lower=0> sigma;         // observation error standard deviation
    vector<lower=0>[K] omega;  // variances for positive_rho_s, rho_r, pi_r

    vector<lower=0>[K] alpha;     // intercepts for positive_rho_s, rho_r, pi_r 
    matrix<lower=0>[K, P] beta; // coefficients beta for positive_rho_s, rho_r, pi_r
}
transformed parameters {
    matrix<lower=0>[N,M] y_hat; // Predicted y values 
    for (i in 1 : N) {
        for (j in 1 : M) {
            y_hat[i,j] = predict_y(N, i, j, psi, rho_r, positive_rho_s, pi_r, t);
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
        positive_rho_s[i] ~ gamma(alpha[1] + beta[2] * x[i]', omega[1]);
    for (i in 1:N)
        rho_r[i] ~ gamma(alpha[2] + beta[2] * x[i]', omega[2]);
    for (i in 1:N)
        pi_r[i] ~ beta(alpha[3] + beta[3] * x[i]', omega[3]);
    for (i in 1:N)
        psi[i] ~ normal(0,1);

    // Likelihood
    for (i in 1 : N) {
        for (j in 1 : M) {
            y[i,j] ~ normal(y_hat[i,j],sigma);
        }
    }
}
