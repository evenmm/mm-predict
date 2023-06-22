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
Y_parameters = [elem.to_prediction_array_composite_g_s_and_K_1() for elem in Y_parameters]
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

def sort_by_test(pred_in, test_in, index):
    # index 0: pi_r. index 1: g_r
    compare = [[pred_in[ii], test_in[ii]] for ii, elem in enumerate(test_in)]
    sorted_compare = Sort(compare)
    pred_array = [elem[0] for elem in sorted_compare]
    truth_array = [elem[1] for elem in sorted_compare]
    return pred_array, truth_array

def make_figure(pred_array, truth_array, name, TRAIN=True):
    plt.figure()
    plt.scatter(range(len(pred_array)), truth_array, c="navy", s=s, edgecolor="black", label="Step 1: Estimate")
    plt.scatter(range(len(pred_array)), pred_array, c="red", s=s, edgecolor="black", label="Step 2: Prediction")
    plt.xlabel("Sorted by true "+name)
    plt.ylabel(name+": Fraction resistant cells")
    plt.legend(loc="best")
    if TRAIN:
        plt.title("Train data: Compare truth and predictions")
        plt.savefig("./plots/diagnostics_train_"+name+"_estimate_compared_to_truth.png")
    else: 
        plt.title("Test data: Compare truth and predictions")
        plt.savefig("./plots/diagnostics_test_"+name+"_estimate_compared_to_truth.png")
    plt.show()
    plt.close()

def train_random_forest(args):
    df_X_covariates, Y_outcome, i = args

    X_full_train, X_full_test, y_train, y_test = train_test_split(df_X_covariates, Y_outcome, test_size=.4, random_state=randomnumberr + i)
    random_forest_model.fit(X_full_train, y_train)
    y_pred = random_forest_model.predict(X_full_test)
    y_pred_train = random_forest_model.predict(X_full_train)
    r2score_train = r2_score(y_train, y_pred_train)
    r2score_test = r2_score(y_test, y_pred)
    return (r2score_train, r2score_test, X_full_train, X_full_test, y_train, y_test, y_pred, y_pred_train)

def estimate_single_parameter_and_plot(parameter_position, variable_name):
    print("\nTraining and testing")
    print("Variable: "+variable_name)
    Y_pi = [elem[parameter_position] for elem in Y_parameters]
    all_random_states = range(50)
    args = [(df_X_covariates, Y_pi, i) for i in all_random_states]
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

    print("Average R2 score train:", np.mean(r2score_train_array), "std:", np.std(r2score_train_array))
    print("Average R2 score test:", np.mean(r2score_test_array), "std:", np.std(r2score_test_array))

    compare_array = [[y_pred[ii], y_test[ii]] for ii, elem in enumerate(y_test)]
    compare_array = Sort(compare_array)
    compare_array_pred = [elem[0] for elem in compare_array]
    compare_array_test = [elem[1] for elem in compare_array]

    pred_array_train, truth_array_train = sort_by_test(y_pred_train, y_train, 0)
    make_figure(pred_array_train, truth_array_train, variable_name, TRAIN=True)

    pred_array_test, truth_array_test = sort_by_test(y_pred, y_test, 0)
    make_figure(pred_array_test, truth_array_test, variable_name, TRAIN=False)

    #########################
    # Save things
    #########################
    picklefile = open('./binaries_and_pickles/random_forest_model_'+variable_name, 'wb')
    pickle.dump(random_forest_model, picklefile)
    picklefile.close()

    picklefile = open('./binaries_and_pickles/X_test_array_learn_mapping_'+variable_name, 'wb')
    pickle.dump(X_full_test_array, picklefile)
    picklefile.close()

    picklefile = open('./binaries_and_pickles/y_test_array_learn_mapping_'+variable_name, 'wb')
    pickle.dump(y_test_array, picklefile)
    picklefile.close()
    return random_forest_model, X_full_test_array, y_test_array

# pi_r
random_forest_model_pi_r, X_full_test_array_pi_r, y_test_array_pi_r = estimate_single_parameter_and_plot(0, "pi_r")
end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)

# g_r
random_forest_model_g_r, X_full_test_array_g_r, y_test_array_g_r = estimate_single_parameter_and_plot(1, "g_r")
end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)

# (g_s - K)
random_forest_model_g_s, X_full_test_array_g_s, y_test_array_g_s = estimate_single_parameter_and_plot(2, "(g_s-K)")
end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)


# Test how the rnadom forest estimated parameters are at predicting the binary outcome
# This is the truth
picklefile = open('./binaries_and_pickles/Y_increase_or_not', 'rb')
Y_increase_or_not = pickle.load(picklefile)
picklefile.close()

# Take the test data, match it case by case to the truth. 
# We only have 0 and 1, so we need a probabilistic model to do this
# Take distribution of parameters
# Get marginal posterior of M protein value after x months. This gives a probability that the value is above Y0. 
# AUC of this probability and the fitted model on the true data 
# If this beats the baseline ML model, then very very nice 


