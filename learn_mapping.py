# Purposes of this script: 
#   Take extracted features and parameter estimates
#   Learn mapping from extracted features to drug response parameters
from utilities import *
start_time = time.time()
warnings.simplefilter("ignore")

# Load period and patient definitions
training_instance_dict = np.load("./binaries_and_pickles/training_instance_dict.npy", allow_pickle=True).item()
training_instance_id_list = [key for key in training_instance_dict.keys()] 
picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

# load Y (parameters)
picklefile = open('./binaries_and_pickles/Y_parameters', 'rb')
Y_parameters = pickle.load(picklefile)
Y_parameters = [elem.to_prediction_array_composite_g_s_and_K_1()[0] for elem in Y_parameters]
Y_parameters = np.array(Y_parameters)
print("Number of intervals:", len(Y_parameters))
#plt.hist(Y_parameters[:,0]) # Half mixture params zero, half nonzero. Interesting! (Must address how sensitive sensitive are too)
#plt.show()
picklefile.close()

# Load df_X_covariates
picklefile = open('./binaries_and_pickles/df_X_covariates', 'rb')
df_X_covariates = pickle.load(picklefile)
picklefile.close()

######################################################################
# Split into train and test 
######################################################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
# Train a random forest regressor
randomnumberr = 4219
random_forest_model = RandomForestRegressor(n_estimators=1000, random_state=randomnumberr)

r2score_train_array = []
r2score_test_array = []
print("Training and testing")

def train_random_forest(args):
    df_X_covariates, Y_outcome, i = args

    X_full_train, X_full_test, y_train, y_test = train_test_split(df_X_covariates, Y_outcome, test_size=.4, random_state=randomnumberr + i)
    random_forest_model.fit(X_full_train, y_train)
    y_pred = random_forest_model.predict(X_full_test)
    y_pred_train = random_forest_model.predict(X_full_train)
    r2score_train = r2_score(y_train, y_pred_train)
    r2score_test = r2_score(y_test, y_pred)
    return (r2score_train, r2score_test, X_full_train, X_full_test, y_train, y_test, y_pred, y_pred_train)

all_random_states = range(50)
args = [(df_X_covariates, Y_parameters, i) for i in all_random_states]
with Pool(15) as pool:
    results = pool.map(train_random_forest, args)


r2score_train_array = [elem[0] for elem in results]
r2score_test_array = [elem[1] for elem in results]
X_full_train_array = [elem[2] for elem in results]
X_full_test_array = [elem[3] for elem in results]
y_train_array = [elem[4] for elem in results]
y_test_array = [elem[5] for elem in results]
y_pred_array = [elem[6] for elem in results]
y_pred_train_array = [elem[7] for elem in results]

y_test = y_test_array[-1]
y_train = y_train_array[-1]
y_pred = y_pred_array[-1]
y_pred_train = y_pred_train_array[-1]

#for ii in range(50):
#    X_full_train, X_full_test, y_train, y_test = train_test_split(df_X_covariates, Y_parameters, test_size=.4, random_state=ii+randomnumberr)
#    random_forest_model.fit(X_full_train, y_train)
#    y_pred = random_forest_model.predict(X_full_test)
#    y_pred_train = random_forest_model.predict(X_full_train)
#    r2score_train_array.append(r2_score(y_train, y_pred_train))
#    r2score_test_array.append(r2_score(y_test, y_pred))

######################################################################
end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)
print("Average R2 score train:", np.mean(r2score_train_array), "std:", np.std(r2score_train_array))
print("Average R2 score test:", np.mean(r2score_test_array), "std:", np.std(r2score_test_array))

# Print true and estimated
#for index, elem in enumerate(y_pred):
#    print(y_test[index][0], ":", y_pred[index][0])
#    print(y_test[index][1], ":", y_pred[index][1])
#    print(y_test[index][2], ":", y_pred[index][2], "\n")

s = 25
"""
plt.figure()
plt.scatter(y_test[:, 0], y_test[:, 1], c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(y_pred[:, 0], y_pred[:, 1], c="red", s=s, edgecolor="black", label="Prediction")
#plt.xlim([-6, 6])
#plt.ylim([-6, 6])
plt.xlabel("pi_R: Fraction resistant cells")
plt.ylabel("g_r")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
#plt.show()
plt.close()
"""

compare_pi_r = [[y_pred[ii], y_test[ii]] for ii, elem in enumerate(y_test)]
"""
# These comparison lists of parameters contain pairs of (prediction, truth), sorted by truth, for plotting 
compare_pi_r = [[y_pred[ii][0], y_test[ii][0]] for ii, elem in enumerate(y_test)]
compare_g_r = [[y_pred[ii][1], y_test[ii][1]] for ii, elem in enumerate(y_test)]
compare_g_s = [[y_pred[ii][2], y_test[ii][2]] for ii, elem in enumerate(y_test)]
"""

def sort_by_test(pred_in, test_in, index):
    # index 0: pi_r. index 1: g_r
    compare = [[pred_in[ii], test_in[ii]] for ii, elem in enumerate(test_in)]
    sorted_compare = Sort(compare)
    pred_array = [elem[0] for elem in sorted_compare]
    test_array = [elem[1] for elem in sorted_compare]
    return pred_array, test_array

compare_pi_r = Sort(compare_pi_r)
compare_pi_r_pred = [elem[0] for elem in compare_pi_r]
compare_pi_r_test = [elem[1] for elem in compare_pi_r]

"""
compare_g_r = Sort(compare_g_r)
compare_g_r_pred = [elem[0] for elem in compare_g_r]
compare_g_r_test = [elem[1] for elem in compare_g_r]

compare_g_s = Sort(compare_g_s)
compare_g_s_pred = [elem[0] for elem in compare_g_s]
compare_g_s_test = [elem[1] for elem in compare_g_s]
"""

def make_figure(pred_array, test_array, name):
    plt.figure()
    plt.scatter(range(len(pred_array)), test_array, c="navy", s=s, edgecolor="black", label="Data")
    plt.scatter(range(len(pred_array)), pred_array, c="red", s=s, edgecolor="black", label="Prediction")
    plt.xlabel("Sorted by true "+name)
    plt.ylabel(name+": Fraction resistant cells")
    plt.title("Train data: Compare truth and predictions")
    plt.legend(loc="best")
    plt.savefig("./plots/diagnostics_train_"+name+"_estimate_compared_to_truth.png")
    plt.show()

# pi_r, g_r, g_s
#for iii in range(3):
pred_array, test_array = sort_by_test(y_pred_train, y_train, 0)
make_figure(pred_array, test_array, "pi_r")
"""
pred_array, test_array = sort_by_test(y_pred_train, y_train, 1)
make_figure(pred_array, test_array, "g_r")
pred_array, test_array = sort_by_test(y_pred_train, y_train, 2)
make_figure(pred_array, test_array, "g_s")
"""

plt.figure()
plt.scatter(range(len(compare_pi_r_test)), compare_pi_r_test, c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(range(len(compare_pi_r_pred)), compare_pi_r_pred, c="red", s=s, edgecolor="black", label="Prediction")
plt.xlabel("Sorted by true pi_R")
plt.ylabel("pi_R: Fraction resistant cells")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.savefig("./plots/diagnostics_pi_R_estimate_compared_to_truth.png")
plt.show()
"""

plt.figure()
plt.scatter(range(len(compare_g_r_test)), compare_g_r_test, c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(range(len(compare_g_r_pred)), compare_g_r_pred, c="red", s=s, edgecolor="black", label="Prediction")
plt.xlabel("Sorted by true g_r")
plt.ylabel("g_r: Growth rate of resistant cells")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.savefig("./plots/diagnostics_g_r_estimate_compared_to_truth.png")
plt.show()

plt.figure()
plt.scatter(range(len(compare_g_s_test)), compare_g_s_test, c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(range(len(compare_g_s_pred)), compare_g_s_pred, c="red", s=s, edgecolor="black", label="Prediction")
plt.xlabel("Sorted by true g_s")
plt.ylabel("g_s: Growth rate of resistant cells")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.savefig("./plots/diagnostics_g_s_estimate_compared_to_truth.png")
plt.show()

# Use latest M protein as Y_0 
#for training_instance_id in training_instance_id_list:
# iterate the patient names in X_full_test:
##for index, row in extracted_features.iterrows():
#    training_instance_id = row['training_instance_id']
#    PUBLIC_ID = row['PUBLIC_ID']
#    patient = COMMPASS_patient_dictionary[PUBLIC_ID]
#
#    # Plot estimated compared to true... parameters? No, M protein values 
#    estimated_parameters = y[training_instance_id]
#    predicted_parameters = y_pred[training_instance_id]
#    savename = "./COMPASS_plot_comparisons/compare_predicted_with_estimated"+patient.name+".png"
#    plot_to_compare_estimated_and_predicted_drug_dynamics(estimated_parameters, predicted_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=patient.name, savename=savename)

"""

picklefile = open('./binaries_and_pickles/random_forest_model', 'wb')
pickle.dump(random_forest_model, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/X_test_array_learn_mapping', 'wb')
pickle.dump(X_full_test_array, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/y_test_array_learn_mapping', 'wb')
pickle.dump(y_test_array, picklefile)
picklefile.close()
