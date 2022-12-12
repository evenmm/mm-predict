from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import aesara.tensor as at
from sample_from_full_model import *
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")

class experiment:
    def __init__(self, true_sigma_obs, N_patients, P, number_of_measurements, psi_prior, N_samples, N_tuning, target_accept, max_treedepth, FUNNEL_REPARAMETRIZATION):
        self.true_sigma_obs = true_sigma_obs
        self.N_patients = N_patients
        self.P = P
        self.number_of_measurements = number_of_measurements
        self.psi_prior = psi_prior
        self.N_samples = N_samples
        self.N_tuning = N_tuning
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
        self.FUNNEL_REPARAMETRIZATION = FUNNEL_REPARAMETRIZATION
        self.name = "M_"+str(number_of_measurements)+"_P_"+str(P)+"_true_sigma_obs_"+str(true_sigma_obs)+"_N_patients_"+str(N_patients)+"_psi_prior_"+psi_prior+"_N_samples_"+str(N_samples)+"_N_tuning_"+str(N_tuning)+"_target_accept_"+str(target_accept)+"_max_treedepth_"+str(max_treedepth)+"_FUNNEL_REPARAMETRIZATION_"+str(FUNNEL_REPARAMETRIZATION)

experiments = [
    experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=True),
    experiment(true_sigma_obs=1, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=10, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=20, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=True),
    experiment(true_sigma_obs=1, N_patients=100, P=2, number_of_measurements=20, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=10, N_patients=100, P=2, number_of_measurements=20, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=30, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=True),
    experiment(true_sigma_obs=1, N_patients=100, P=2, number_of_measurements=30, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=10, N_patients=100, P=2, number_of_measurements=30, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=40, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=True),
    experiment(true_sigma_obs=1, N_patients=100, P=2, number_of_measurements=40, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=10, N_patients=100, P=2, number_of_measurements=40, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=50, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=True),
    experiment(true_sigma_obs=1, N_patients=100, P=2, number_of_measurements=50, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    experiment(true_sigma_obs=10, N_patients=100, P=2, number_of_measurements=50, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=20, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=20, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=30, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=30, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=40, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=40, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=50, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=50, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=20, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=20, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=30, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=30, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.9, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.9, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="normal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    #experiment(true_sigma_obs=0.1, N_patients=100, P=2, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    ##experiment(true_sigma_obs=0.1, N_patients=100, P=6, number_of_measurements=50, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    ##experiment(true_sigma_obs=0.1, N_patients=100, P=6, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    ##experiment(true_sigma_obs=0.1, N_patients=100, P=6, number_of_measurements=10, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    ##experiment(true_sigma_obs=0.1, N_patients=100, P=6, number_of_measurements=5, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    ##experiment(true_sigma_obs=0.1, N_patients=100, P=6, number_of_measurements=4, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
    ##experiment(true_sigma_obs=0.1, N_patients=100, P=6, number_of_measurements=3, psi_prior="lognormal", N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, FUNNEL_REPARAMETRIZATION=False),
]

def run_experiment(experiment):
    true_sigma_obs = experiment.true_sigma_obs
    N_patients = experiment.N_patients
    P = experiment.P
    number_of_measurements = experiment.number_of_measurements
    name = experiment.name
    psi_prior = experiment.psi_prior
    N_samples = experiment.N_samples
    N_tuning = experiment.N_tuning
    target_accept = experiment.target_accept
    max_treedepth = experiment.max_treedepth
    FUNNEL_REPARAMETRIZATION = experiment.FUNNEL_REPARAMETRIZATION
    print("Running "+name)
    ##############################
    # Generate data
    # True parameter values
    X_mean = np.repeat(0,P)
    X_std = np.repeat(0.5,P)
    X = np.random.normal(X_mean, X_std, size=(N_patients,P))
    X = pd.DataFrame(X, columns = ["Covariate "+str(ii+1) for ii in range(P)])
    # These are the true parameters for x1 = 0 (median):
    rho_s_population = -0.005
    rho_r_population = 0.001
    pi_r_population = 0.4
    psi_population = 50
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_omega = np.array([0.05, 0.10, 0.15])
    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    true_beta_rho_s[0] = 0.8
    true_beta_rho_s[1] = 0.9
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 1.0
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.0
    true_beta_pi_r[1] = 1.1

    print("true_alpha[0]:", true_alpha[0])
    print("true_alpha[1]:", true_alpha[1])
    print("true_alpha[2]:", true_alpha[2])
    print("true_beta_rho_s: ", true_beta_rho_s)
    print("true_beta_rho_r: ", true_beta_rho_r)
    print("true_beta_pi_r: ", true_beta_pi_r)

    days_between_measurements = int(1500/number_of_measurements)
    measurement_times = days_between_measurements * np.linspace(0, number_of_measurements-1, number_of_measurements)
    treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s), (N_patients,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients,1))

    # Patient specific noise / deviation from X effects
    true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
    true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
    true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])
    # To generate the data, we employ a "fourth omega" for psi. But since we do not explain theta_psi by a linear predictor
    #    , we instead estimate xi in the MCMC, which is the standard deviation of psi_i^0 from y_i1. 
    true_omega_for_psi = 0.1
    true_theta_psi = np.random.normal(np.log(psi_population), true_omega_for_psi, size=N_patients)
    print("true_theta_rho_s[0:5]:\n", true_theta_rho_s[0:5])
    print("true_theta_rho_r[0:5]:\n", true_theta_rho_r[0:5])
    print("true_theta_pi_r[0:5]:\n", true_theta_pi_r[0:5])
    print("true_theta_psi[0:5]:\n", true_theta_psi[0:5])

    true_rho_s = - np.exp(true_theta_rho_s)
    true_rho_r = np.exp(true_theta_rho_r)
    true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
    true_psi = np.exp(true_theta_psi)
    patient_dictionary = {}
    for training_instance_id in range(N_patients):
        psi_patient_i   = true_psi[training_instance_id]
        pi_r_patient_i  = true_pi_r[training_instance_id]
        rho_r_patient_i = true_rho_r[training_instance_id]
        rho_s_patient_i = true_rho_s[training_instance_id]
        these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=true_sigma_obs)
        this_patient = Patient(these_parameters, measurement_times, treatment_history, name=str(training_instance_id))
        patient_dictionary[training_instance_id] = this_patient
        #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id))

    #print("Y:\n", Y)
    #print("t:\n", t)
    #print("yi0:\n", yi0)
    #print("X:\n", X)
    print("Done generating data")
    idata = sample_from_full_model(X, patient_dictionary, name, N_samples=N_samples, N_tuning=N_tuning, target_accept=target_accept, psi_prior=psi_prior, max_treedepth=max_treedepth, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION)

    print("Done sampling")
    lines = [('alpha', {}, true_alpha), ('beta_rho_s', {}, true_beta_rho_s), ('beta_rho_r', {}, true_beta_rho_r), ('beta_pi_r', {}, true_beta_pi_r), ('omega', {}, true_omega), ('sigma_obs', {}, true_sigma_obs)]
    az.plot_trace(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma_obs'), lines=lines, combined=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_posterior_group_parameters.png")
    #plt.show()
    plt.close()

    if psi_prior=="lognormal":
        az.plot_trace(idata, var_names=('xi'), combined=True)
        plt.savefig("./plots/posterior_plots/"+name+"-plot_posterior_group_parameters_xi.png")
        #plt.show()
        plt.close()

    lines = [('theta_rho_s', {}, true_theta_rho_s), ('theta_rho_r', {}, true_theta_rho_r), ('theta_pi_r', {}, true_theta_pi_r), ('rho_s', {}, true_rho_s), ('rho_r', {}, true_rho_r), ('pi_r', {}, true_pi_r)]
    az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), lines=lines, combined=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_posterior_individual_parameters.png")
    #plt.show()
    plt.close()
    # Test of exploration 
    az.plot_energy(idata)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_energy.png")
    #plt.show()
    plt.close()
    # Plot of coefficients
    az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_alpha.png")
    plt.tight_layout()
    #plt.show()
    az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True, rope=(0,0))
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_beta_rho_s.png")
    plt.tight_layout()
    #plt.show()
    plt.close()
    az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True, rope=(0,0))
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_beta_rho_r.png")
    plt.tight_layout()
    #plt.show()
    plt.close()
    az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True, rope=(0,0))
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_beta_pi_r.png")
    plt.tight_layout()
    #plt.show()
    plt.close()
    az.plot_forest(idata, var_names=["theta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_theta_rho_s.png")
    plt.tight_layout()
    #plt.show()
    plt.close()
    az.plot_forest(idata, var_names=["theta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_theta_rho_r.png")
    plt.tight_layout()
    #plt.show()
    plt.close()
    az.plot_forest(idata, var_names=["theta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig("./plots/posterior_plots/"+name+"-plot_forest_theta_pi_r.png")
    plt.tight_layout()
    #plt.show()
    plt.close()

    try: 
        az.plot_posterior(idata, var_names="tree_depth", group="sample_stats")
        plt.savefig("./plots/posterior_plots/"+name+"-plot_tree_depth.png")
        #plt.show()
        plt.close()
    except:
        print("Couldn't plot posterior of tree depth")

for elem in experiments: 
    run_experiment(elem)
