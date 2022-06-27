# Purposes of this script: 
#   Load COMMPASS_patient_dictionary
#   Find sections with the right drug combination and enough data to perform inference
#   Perform inference of parameter set in each region
from utilities import *
from sklearn.linear_model import LinearRegression

# Settings
N_iter = 1000 # separate minimzations of the least squares when fitting parameters
minimum_number_of_measurements = 3 # required number of M protein measurements for a period to be included in the dataset
threshold_for_closeness_for_M_protein_at_start = 60 # If M protein at period start is missing, it is imputed using the nearest measurement, but only if closer than this threshold number of days.
INCLUDE_SUBSEQUENT_DRUG_HOLIDAY = True # If a treatment is followed by a drug holiday, this decided if the holiday is included as part of the period

## Inference
# The length of ub and lb implicitly decides whether the effect of treatment is given a parameter or not. 
# If len(ub) = 4, then it is assumed that only periods under treatment are considered
# If len(ub) = 5, then k_1 models the effect of the drug on the sensitive population
# Simple exponential growth model with 2 populations, where only one is affected by treatment
# The parameters we estimate are 
#               Y_0, pi_r,   g_r,   g_s,  k_1,  sigma
lb = np.array([  0,    0,  0.00,   0.00, 0.20]) #, 10e-6])
ub = np.array([100,    1,  0.20,  lb[4], 1.00]) #, 10e4])
#lb = np.array([  0,    0,  0.00, -1e-0])
#ub = np.array([100,    1,  2e-1,  0.00])
# Y_0=50, pi_r=0.10, g_r=2e-3, g_s=1e-2, k_1=3e-2


# Load COMMPASS_patient_dictionary
picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/unique_treat_counter', 'rb')
unique_treat_counter = pickle.load(picklefile)
picklefile.close()

# A training instance is a pair of history covariates X and estimated parameters Y
# Define minimum number of measurements for including period as training instance to X and Y
print("\nFinding right regions and estimating parameters...")
# Iterate through patients
# Identify valid periods and estimate parameters there: all_drug_holiday_measurements = (training_instance_id, start, end, treatment_id). Y: (training_instance_id, parameters). 
# Find patients with periods that satisfy criteria of enough M protein measurements under chosen treatment 
# 1) Feature extract history up until period, put in X
# 2) Estimate parameters in period, put in Y
training_instance_id = 0
all_drug_holiday_measurements = {} # A dictionary mapping training_instance_id to the patient name and the start and end of the interval with the treatment of interest 
log_growth_rates = []
patient_id_array = []
holiday_number_array = []

# Iterate over all patients, look at their treatment periods one by one and find drug holidays
start_time = time.time()
how_many_regions = np.zeros(unique_treat_counter)
for name, patient in COMMPASS_patient_dictionary.items():
    holiday_counter = 0
    if len(patient.measurement_times) > minimum_number_of_measurements:
        for index, treatment in enumerate(patient.treatment_history): # Outer loop so we pass each of them only once 
            if treatment.id == 0: # drug holidays
                holiday_counter = holiday_counter + 1
                treatment_id_of_interest = treatment.id

                # We found a drug holiday
                period_start = treatment.start
                period_end = treatment.end
                this_history = np.array([treatment])

                dummy_measurement_times = patient.measurement_times
                dummy_Mprotein_values = patient.Mprotein_values
                dummy_Kappa_values = patient.Kappa_values
                dummy_Lambda_values = patient.Lambda_values

                valid_Mprotein = dummy_Mprotein_values[dummy_measurement_times>=period_start]
                valid_Kappa = dummy_Kappa_values[dummy_measurement_times>=period_start]
                valid_Lambda = dummy_Lambda_values[dummy_measurement_times>=period_start]
                valid_times = dummy_measurement_times[dummy_measurement_times>=period_start]
                valid_Mprotein = valid_Mprotein[valid_times<=period_end]
                valid_Kappa = valid_Kappa[valid_times<=period_end]
                valid_Lambda = valid_Lambda[valid_times<=period_end]
                valid_times = valid_times[valid_times<=period_end]

                # Only add as data instance to X and Y if there are enough:
                if len(valid_times) >= minimum_number_of_measurements and max(valid_Mprotein) > 0:
                    # Take away leading zeroes
                    zero_positions = np.where(valid_Mprotein == 0)[-1]
                    if min(valid_Mprotein) <= 0:
                        if len(zero_positions) > 0:
                            print(zero_positions)
                            last_zero_position = zero_positions[-1]
                            print(last_zero_position)
                            valid_times = valid_times[last_zero_position:]
                            valid_Mprotein = valid_Mprotein[last_zero_position:]
                    
                    if len(valid_times) >= minimum_number_of_measurements and max(valid_Mprotein) > 0:
                        print("Saving a case from", patient.name, "- treatment id", treatment_id_of_interest)
                        # Note the time limits of this period
                        all_drug_holiday_measurements[training_instance_id] = [patient.name, period_start, period_end, treatment_id_of_interest, valid_times, valid_Mprotein]
                        # Estimate log growth rate
                        valid_times = valid_times.reshape((-1, 1))
                        model = LinearRegression()
                        valid_Mprotein = valid_Mprotein + 1e-15
                        log_valid_Mprotein = np.log(valid_Mprotein)
                        model.fit(valid_times, log_valid_Mprotein)
                        growth_rate = model.coef_[0]
                        log_growth_rates.append(growth_rate) # training_instance_id is position in log_growth_rates
                        patient_id_array.append(patient.name)
                        holiday_number_array.append(holiday_counter)
                        #log_growth_rates = np.concatenate((log_growth_rates, growth_rate)) # training_instance_id is position in log_growth_rates
                        training_instance_id = training_instance_id + 1

#https://stackoverflow.com/questions/56638467/how-to-fill-color-by-groups-in-histogram-using-matplotlib
all_log_growth_rates = np.array(log_growth_rates)
all_patient_id_array = np.array(patient_id_array)
holiday_number_array = np.array(holiday_number_array)
print(all_log_growth_rates)
print(all_patient_id_array)

# Histograms of growth rates
# Colored per patient
# The first 61 items, grouped by patients
log_growth_rates = all_log_growth_rates[0:61]
patient_id_array = all_patient_id_array[0:61]
data = {'log_growth_rates': log_growth_rates,
        'Patient': patient_id_array}
df = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=df, x='log_growth_rates', hue='Patient', stat='count', edgecolor=None, bins=200, legend=False, palette=drug_colors)
ax.set_title('log_growth_rates')
plt.savefig("./plots/histogram_per_patient_part_1.png")
plt.show()
# The rest of the regions, grouped by patients
log_growth_rates = all_log_growth_rates[61:]
patient_id_array = all_patient_id_array[61:]
data = {'log_growth_rates': log_growth_rates,
        'Patient': patient_id_array}
df = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=df, x='log_growth_rates', hue='Patient', stat='count', edgecolor=None, bins=200, legend=False, palette=drug_colors)
ax.set_title('log_growth_rates')
plt.savefig("./plots/histogram_per_patient_part_2.png")
plt.show()

# Histograms of growth rates
# Colored per drug holiday
data = {'log_growth_rates': all_log_growth_rates,
        'Holiday_number': holiday_number_array}
df = pd.DataFrame(data)
all_holiday_numbers = pd.unique(df[['Holiday_number']].values.ravel('K'))
N_different_holidays = len(all_holiday_numbers)
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=df, x='log_growth_rates', hue='Holiday_number', stat='count', edgecolor=None, bins=200, legend=True, palette=sns.color_palette("viridis", n_colors=N_different_holidays)) #sns.color_palette("husl", N_different_holidays))
ax.set_title('log_growth_rates')
plt.savefig("./plots/histogram_per_holiday_number.png")
plt.show()

log_growth_rates = all_log_growth_rates
fig, ax = plt.subplots()
meadian_growth_rate = np.median(log_growth_rates)
avg_growth_rate = np.mean(log_growth_rates)
std_growth_rate = np.std(log_growth_rates)
#"Median: "+f'{meadian_growth_rate:.3g}'
labelllll = "Mean:      "+f'{avg_growth_rate:.3g}'+"\nStd:         "+f'{std_growth_rate:.3g}'+"\nN = "+str(len(log_growth_rates))
ax.hist(log_growth_rates, bins=100) # Half mixture params zero, half nonzero. Interesting! (Must address how sensitive sensitive are too)
ax.axvline(np.mean(log_growth_rates), color="k", linewidth=0.5, linestyle="-", label=labelllll)
ax.set_title("Histogram of growth rates in drug holidays")
ax.set_xlabel("Growth rate")
ax.set_ylabel("Number of intervals")
ax.legend()
plt.savefig("./plots/drug_holiday_histogram_of_growth_rates.png")
#plt.show()

print(all_log_growth_rates)
drug_colors
fig, ax = plt.subplots()
meadian_growth_rate = np.median(all_log_growth_rates)
avg_growth_rate = np.mean(all_log_growth_rates)
std_growth_rate = np.std(all_log_growth_rates)
#"Median: "+f'{meadian_growth_rate:.3g}'
labelllll = "Mean:      "+f'{avg_growth_rate:.3g}'+"\nStd:         "+f'{std_growth_rate:.3g}'+"\nN = "+str(len(all_log_growth_rates))
ax.hist(all_log_growth_rates, bins=100) # Half mixture params zero, half nonzero. Interesting! (Must address how sensitive sensitive are too)
ax.axvline(np.mean(all_log_growth_rates), color="k", linewidth=0.5, linestyle="-", label=labelllll)
ax.set_title("Histogram of growth rates in drug holidays")
ax.set_xlabel("Growth rate")
ax.set_ylabel("Number of intervals")
ax.legend()
plt.savefig("./plots/drug_holiday_histogram_of_growth_rates.png")
#plt.show()

fig, ax = plt.subplots()
for key, value in all_drug_holiday_measurements.items():
    valid_times = value[4]
    valid_Mprotein = value[5]
    #valid_Mprotein = valid_Mprotein + 1e-15
    log_valid_Mprotein = np.log(valid_Mprotein)
    plt.plot(valid_times, log_valid_Mprotein, color="k", linewidth=0.5, linestyle="-", marker="x")
ax.set_title("Log transformed M protein values in drug holidays")
ax.set_xlabel("Days")
ax.set_ylabel("Serum Mprotein (g/dL)")
plt.savefig("./plots/drug_holiday_Mproteins_logscale.png")
#plt.show()

fig, ax = plt.subplots()
for key, value in all_drug_holiday_measurements.items():
    valid_times = value[4]
    valid_Mprotein = value[5]
    plt.plot(valid_times, valid_Mprotein, color="k", linewidth=0.5, linestyle="-", marker="x")
ax.set_title("M protein values in drug holidays")
ax.set_xlabel("Days")
ax.set_ylabel("Serum Mprotein (g/dL)")
plt.savefig("./plots/drug_holiday_Mproteins.png")
#plt.show()

end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)
print("Number of intervals:", len(all_log_growth_rates))
