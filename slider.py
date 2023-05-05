from utilities import *
from matplotlib.widgets import Slider, Button, RadioButtons
RANDOM_SEED = 499
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
PLOT_RESISTANT = True
FUNNEL_REPARAMETRIZATION = False
MODEL_RANDOM_EFFECTS = True
N_HIDDEN = 2
RANDOM_EFFECTS = True
P = 5 # Number of covariates
P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
true_omega = np.array([0.10, 0.05, 0.20])
true_omega_for_psi = 0.1
true_sigma_obs = 0
M_number_of_measurements = 12
N_patients = 150
y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
max_time = 180
days_between_measurements = int(max_time/M_number_of_measurements)
measurement_times = days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, get_expected_theta_from_X_2, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS)

# Our patient 
ii = 19
patient = patient_dictionary[ii]
parameters = parameter_dictionary[ii]
measurement_times = patient.measurement_times
treatment_history = patient.get_treatment_history()
parameters = Parameters(Y_0=50, pi_r=0.1*parameters.pi_r, g_r=30*parameters.g_r, g_s=30*parameters.g_s, k_1=0, sigma=true_sigma_obs)
patient = Patient(parameters, measurement_times, treatment_history, name=str(ii))
plot_mprotein(patient, "", "./obs.pdf", PLOT_PARAMETERS=True, parameters=parameters)
#Mprotein_values = patient.get_Mprotein_values()
time_zero = min(treatment_history[0].start, measurement_times[0])
time_max = find_max_time(measurement_times)
plotting_times = np.linspace(time_zero, time_max, y_resolution)


################# sliders ###############
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.25)
mprot = measure_Mprotein_with_noise(parameters, plotting_times, treatment_history)
#parameters = Parameters(50, 0.072, -0.056, 0.01742, parameters.k_1, parameters.sigma)
resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
mres = measure_Mprotein_with_noise(resistant_parameters, plotting_times, treatment_history)
msens = mprot - mres
l, = plt.plot(plotting_times, mprot, lw=2, color='k', label="From all cells", zorder=3)
r, = plt.plot(plotting_times, mres, lw=2, color='r', linestyle="--", label="From resistant", zorder=2)
s, = plt.plot(plotting_times, msens, lw=2, color='b', linestyle="--", label="From sensitive", zorder=1)
plt.ylabel("Serum Mprotein (g/L)")
plt.axis([plotting_times[0], plotting_times[-1], 0, 80])
plt.xlabel("Days")
plt.legend()
axcolor = 'lightgoldenrodyellow'
ax_pi    = plt.axes([0.25, 0.12, 0.65, 0.03], facecolor=axcolor)
ax_rho   = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
ax_alpha = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)
#ax_Y_0 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

slide_pi    = Slider(ax_pi, r'$\pi $', 0, 1, valinit=parameters.pi_r[0])
slide_rho   = Slider(ax_rho, r'$\rho $', 0, 10*parameters.g_r[0], valinit=parameters.g_r[0])
slide_alpha = Slider(ax_alpha, r'$\alpha$', - 0.1, 0, valinit=parameters.g_s[0])
#slide_Y_0 = Slider(ax_Y_0, 'Y_0', - 0.1, 0, valinit=parameters.Y_0[0])

def update(val):
    pi = slide_pi.val
    rho = slide_rho.val
    alpha = slide_alpha.val
    #Y_0 = slide_Y_0.val
    params = Parameters(parameters.Y_0, pi, rho, alpha, parameters.k_1, parameters.sigma)
    r_params = Parameters((params.Y_0*params.pi_r), 1, params.g_r, params.g_s, params.k_1, params.sigma)
    mprot = measure_Mprotein_with_noise(params, plotting_times, treatment_history)
    mres = measure_Mprotein_with_noise(r_params, plotting_times, treatment_history)
    msens = mprot - mres
    l.set_ydata(mprot)
    r.set_ydata(mres)
    s.set_ydata(msens)
    fig.canvas.draw_idle()
slide_pi.on_changed(update)
slide_rho.on_changed(update)
slide_alpha.on_changed(update)

plt.show()
