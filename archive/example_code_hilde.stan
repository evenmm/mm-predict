
  // Hilde comments: Use this block to define functions that are fequently used throughout the rest of the script.
functions {
  // Function to derive the likelihood of weekly hospitalization counts
  // given: number of new infections on each day (lambda_hosp_day)
  // probability of hospitalizationfor every day of infection (p_hosp_days)
  // distribution of time between infection and hospitalization (tau_hosp)
  vector likel_hosp_week(int n_days, int n_weeks, int hosp_cutoff,
                         vector tau_hosp, vector new_inf_days,
                         vector p_hosp_days) {
    vector[n_days] lambda_hosp_day;
    vector[n_weeks] lambda_hosp_week;
    for (t in 1 : n_days) {
      lambda_hosp_day[t] = sum((new_inf_days[max(1,
                                                 t - hosp_cutoff) : t])
                               .* (p_hosp_days[max(1,
                                                   t - hosp_cutoff) : t])
                               .* reverse(tau_hosp[1 : min(t, hosp_cutoff+1)]));
    }
    for (w in 1 : n_weeks) {
      lambda_hosp_week[w] = sum(lambda_hosp_day[(1 + ((w - 1) * 7)) : (
                                7 + ((w - 1) * 7))]);
    }
    return (lambda_hosp_week);
  }
  // rep_each function
  vector rep_each(vector x, int K) {
    int N = rows(x);
    vector[N * K] y;
    int pos = 1;
    for (n in 1 : N) {
      for (k in 1 : K) {
        y[pos] = x[n];
        pos += 1;
      }
    }
    return y;
  }
}

// Hilde comments: The dimensions we have are regions, time and number of covariates. 
data {
  // SIR model and associated data
  int<lower=1> n_days; // number of days of covariates
  int<lower=1> n_regions; // number of analysed regions
  int<lower=1> n_groups; // number of groups of regions (random-walk of beta)
  array[n_regions] int<lower=1, upper=n_groups> group_region;
  int<lower=1> n_col; // number columns/covariates design matrix
  array[n_regions] matrix[n_days,n_col] covariates; // Designmatrix without intercept for the region specific parameter
  int<lower=1> n_var; // number of considered variants (2 for alpha and delta)
  array[n_regions] matrix[n_days, n_var] variants;
  array[n_regions] int<lower=1> n_agegroup; // number of age-groups per region used for accessing relevant 
  int n_agegroup_max; // maximum number of age-groups, used to specify padded objects containing -Inf for irrelevant values
  array[n_regions] vector[n_agegroup_max] pop_r_age; // population size per region and age-group at baseline, padded object with -Inf for irrelevant age-groups
  array[n_regions] matrix[n_days, n_agegroup_max] n_vaccinated; // number of vaccinated individuals age age-group and region, padded object with -Inf for irrelevant age-groups
  array[n_regions] vector<upper=1>[n_agegroup_max] p_hosp_age; // hospitalization risk per age-group in region, padded object with -Inf for irrelevant age-groups
  real<lower=0> gamma;
  int<lower=0> hosp_cutoff; // maximum number of days between infection and hosp. in days
  vector<lower=0>[hosp_cutoff+1] tau_hosp; // distribution of time between infection and hospitalizations.
  // Weekly data
  int<lower=1> n_weeks; // Number of weeks
  array[n_regions, n_weeks] int hosp_data; // Weekly hospital data. One row per region
  // Prior params
  //real<lower=0.5> sd_phi;
}


parameters {
  vector<lower=0.5, upper=10>[n_regions] r_0;
  array[n_groups, n_weeks-1] real phi_raw;
  vector<lower=0>[n_groups] sd_phi_week;
  vector[n_col] phi_cov;
  vector[n_var] phi_var;
}

// Hilde comments: We are essentially interested in phi_cov (above), phi_var (above) and phi_week (which kinda corresponds to your beta). 
transformed parameters {
  array[n_groups] vector[n_weeks] phi_week; // weekly random-walk intercepts in groups of regions

  array[n_regions] matrix[n_days, n_agegroup_max] S; // the susceptibles in each region and age-group (padded)
  array[n_regions] matrix[n_days, n_agegroup_max] I; // infectious in each region and age-group (padded)
  array[n_regions] matrix[n_days, n_agegroup_max] R_n; // removed in each region and age-group (padded)
  array[n_regions] vector[n_days] R_v; // removed by vaccination
  
  array[n_regions] vector[n_days] beta; //The regression line of beta
  array[n_regions] vector<lower=0>[n_days] new_inf_days; // New infections per day in region
  array[n_regions] vector<lower=0>[n_days] p_hosp_days; // Hospitalization prob. among infected per day in region
  
  array[n_regions] vector<lower=0>[n_weeks] lambda_hosp_week;
  
  // Group-specific weekly random walk
  for (g in 1:n_groups) {
    phi_week[g][1] = 0;
    for (w in 2:n_weeks) {
      phi_week[g][w] = sd_phi_week[g]*phi_raw[g, w-1];
    }
  }
  
  // Hilde comments: This is the regression! Our beta corresponds in a way to your psi.
  
  // The regression of beta.
  for (r in 1 : n_regions) {
    beta[r] = exp(log(r_0[r] * gamma) + covariates[r]*phi_cov + variants[r]*phi_var + rep_each(phi_week[group_region[r],], 7));
  }
  
  // Hile comments: We have a deterministic SIR model that we run over each day and each region.

  //initialization (need to be at least one infectious individuals in each region at time t=0)
  //0.5 per 100000 of the population in each region is initially infectious
  for (r in 1 : n_regions) {
    for (j in 1 : n_agegroup[r]) {
      I[r, 1, j] = pop_r_age[r, j] / 100000 * 0.5; // 0.5 per 100000 of population in region and age-groups
      S[r, 1, j] = pop_r_age[r, j] - I[r, 1, j]; // Initial number of susceptibles per age-group
      R_n[r, 1, j] = 0; // no removed 
    }
    R_v[r, 1] = 0;
  }
  
  // The SIR model
  profile("SIR-trans_data") {
  for (r in 1 : n_regions) {
    for (t in 2 : n_days) {
      for (j in 1 : n_agegroup[r]) {
        // Susceptibles at time t: Susceptibles at t-1 minus new infections, minus vaccinations among susceptibles, minus "imported" infections
        S[r, t, j] = S[r, t - 1, j]
                     - ((beta[r, t] ./ sum(pop_r_age[r, 1 : n_agegroup[r]]))
                        .* sum(I[r, t - 1, 1 : n_agegroup[r]])
                        .* S[r, t - 1, j])
                     - (n_vaccinated[r, t - 1, j]
                        .* (S[r, t - 1, j]
                            ./ (S[r, t - 1, j] + R_n[r, t - 1, j])))
                     - (S[r, t - 1, j] / 100000 * 0.5); // "Importation": 0.5 per 100000 susceptibles
        // Infections at time t: Infections at t-1 + new infected, minus recovered, plus "imported" infections  
        I[r, t, j] = I[r, t - 1, j]
                     + ((beta[r, t] ./ sum(pop_r_age[r, 1 : n_agegroup[r]]))
                        .* sum(I[r, t - 1, 1 : n_agegroup[r]])
                        .* S[r, t - 1, j])
                     - (gamma * I[r, t - 1, j])
                     + (S[r, t - 1, j] / 100000 * 0.5); // "Importation": 0.5 per 100000 susceptibles
        // Recovered natural at time t: Recovered natural at t-1, plus newly recovered among infected, minus vaccinated among recovered    
        R_n[r, t, j] = R_n[r, t - 1, j] + (gamma * I[r, t - 1, j])
                       - (n_vaccinated[r, t - 1, j]
                          .* (R_n[r, t - 1, j]
                              ./ (S[r, t - 1, j] + R_n[r, t - 1, j])));
      }
      // Recovered vaccinated at time t: Recovered vaccinated at t-1 plus new number of vaccinated.
      R_v[r, t] = R_v[r, t - 1]
                  + sum(n_vaccinated[r, t - 1, 1 : n_agegroup[r]]);
    }
  }
  }
  // Hilde comments: We need to calculate some quantities later used in hte likelihood.
  
  // Derive quantities from latent compartments
  profile("new_inf_p_hosp-trans_data") {
  for (r in 1 : n_regions) {
    // New infections from one day to the other
    new_inf_days[r, 1] = sum(I[r, 1, 1 : n_agegroup[r]]);
    // Hospitalization probability at day t
    p_hosp_days[r, 1] = sum(pop_r_age[r][1 : n_agegroup[r]]
                            .* p_hosp_age[r][1 : n_agegroup[r]])
                        / sum(pop_r_age[r][1 : n_agegroup[r]]);
    for (t in 2 : n_days) {
      // New infections: Difference in susceptibles yesterday and today (>0), minus vaccinations among susceptibles
      new_inf_days[r, t] = sum(S[r, t - 1, 1 : n_agegroup[r]])
                           - sum(S[r, t, 1 : n_agegroup[r]])
                           - (sum(n_vaccinated[r, t - 1, 1 : n_agegroup[r]]
                                  .* (S[r, t - 1, 1 : n_agegroup[r]]
                                      ./ (S[r, t - 1, 1 : n_agegroup[r]]
                                          + R_n[r, t - 1, 1 : n_agegroup[r]]))));
      // Hospitalisation probability among infected on day t: weighted sum of hospitalization probabilities in age groups by share of age 
      // groups among susceptibles at t-1. Considered variants (alpha and delta) increase hosp. prob by factor 1.9   
      p_hosp_days[r, t] = ((S[r, t - 1, 1 : n_agegroup[r]]
                            * p_hosp_age[r][1 : n_agegroup[r]])
                           / sum(S[r, t - 1, 1 : n_agegroup[r]]))
                          * (1 - sum(variants[r, t - 1,  : ])
                             + sum(variants[r, t - 1,  : ]) * 1.9);
    }
  }
  }
  
  // Hilde comments: Same here, this is used in the likelihood further down
  profile("likelihood-trans_data") {
    // Hospitalization likelihood
    for (r in 1 : n_regions) {
      lambda_hosp_week[r] = likel_hosp_week(n_days, n_weeks, hosp_cutoff,
      tau_hosp, new_inf_days[r],
      p_hosp_days[r]);
    }
  }
}
model {
  // Prior distributions
  profile("sampling-prior_r_0_sd_phi") {
    r_0 ~ normal(2, 0.5);
    sd_phi_week ~ normal(0, .25);
  }
  profile("sampling-effects") {
    phi_cov ~ normal(0, 0.5);
    phi_var ~ normal(0, 0.5);
    to_array_1d(phi_raw) ~ std_normal();
  }
  //Likelihoods hospitalizations
  profile("sampling-likelihood") {
    for (r in 1 : n_regions) {
      hosp_data[r, ] ~ poisson(lambda_hosp_week[r]);
    }
  }
}
