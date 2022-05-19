# Take history regions and estimates
# Extract features and learn the mapping from features to drug response parameters
from utilities import *
import warnings
warnings.simplefilter("ignore")

# Load period definitions
training_instance_dict = np.load("training_instance_dict.npy", allow_pickle=True).item()
training_instance_id_list = [key for key in training_instance_dict.keys()] 
df_mprotein_and_dates = pd.read_pickle("df_mprotein_and_dates.pkl")
df_drugs_and_dates = pd.read_pickle("df_drugs_and_dates.pkl")
#print(df_mprotein_and_dates.head(n=5))

# load Y (parameters)
picklefile = open('Y_parameters', 'rb')
Y_parameters = pickle.load(picklefile)
Y_parameters = [elem.to_prediction_array_composite_g_s_and_K_1()[0] for elem in Y_parameters]
Y_parameters = np.array(Y_parameters)
#print(Y_parameters)
#plt.hist(Y_parameters[:,0]) # Half mixture params zero, half nonzero. Interesting! (Must address how sensitive sensitive are too)
#plt.show()
picklefile.close()

picklefile = open('COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

######################################################################
# tsfresh feature extraction from numerical columns in the dataframes 
######################################################################
# training_instance_dict is used to index into M protein and drug data frames

# M protein
# Subsets taken from df_mprotein_and_dates and labeled by training_instance_id for tsfresh feature extraction 
df_Mprotein_pre_tsfresh = pd.DataFrame(columns=["D_LAB_serum_m_protein", "VISITDY", "training_instance_id"])
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    period_start = value[1]
    period_end = value[2]
    
    single_entry = df_mprotein_and_dates[(df_mprotein_and_dates["PUBLIC_ID"] == patient_name) & (df_mprotein_and_dates["VISITDY"] > period_start) & (df_mprotein_and_dates["VISITDY"] < period_end)]
    single_entry["training_instance_id"] = training_instance_id
    df_Mprotein_pre_tsfresh = pd.concat([df_Mprotein_pre_tsfresh, single_entry])
#print(df_Mprotein_pre_tsfresh.head(n=20))

# Drugs
df_drugs_pre_tsfresh = pd.DataFrame(columns=['PUBLIC_ID', 'MMTX_THERAPY', 'drug_id', 'startday', 'stopday', "training_instance_id"])
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    period_start = value[1]
    period_end = value[2]
    
    single_entry = df_drugs_and_dates[(df_drugs_and_dates["PUBLIC_ID"] == patient_name) & (df_drugs_and_dates["stopday"] <= period_end)]
    single_entry["training_instance_id"] = training_instance_id
    df_drugs_pre_tsfresh = pd.concat([df_drugs_pre_tsfresh, single_entry])
#print(df_drugs_pre_tsfresh.head(n=20))
# Add zero row if the id is missing from the drug df
#for training_instance_id in training_instance_id_list: 
#    if training_instance_id not in pd.unique(df_drugs_pre_tsfresh[['training_instance_id']].values.ravel('K')):
#        dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'drug_id':[0],'startday':[0], 'stopday':[0]})
#        df_drugs_pre_tsfresh = pd.concat([df_drugs_pre_tsfresh, dummy], ignore_index=True)

# Feature extraction requires only numerical values in columns
from tsfresh import extract_features
df_Mprotein_pre_tsfresh = df_Mprotein_pre_tsfresh[["D_LAB_serum_m_protein", "VISITDY", "training_instance_id"]] # Remove PUBLIC_ID
extracted_features_M_protein = extract_features(df_Mprotein_pre_tsfresh, column_id="training_instance_id", column_sort="VISITDY")

df_drugs_pre_tsfresh = df_drugs_pre_tsfresh[['drug_id','startday', 'stopday', "training_instance_id"]] # Remove PUBLIC_ID
extracted_features_drugs = extract_features(df_drugs_pre_tsfresh, column_id="training_instance_id", column_sort="startday")
###################################################################

# Impute nan and inf
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features_M_protein)
impute(extracted_features_drugs)

# Merge drug and M protein data 
extracted_features_tsfresh = extracted_features_drugs.join(extracted_features_M_protein) #, how="outer", on="training_instance_id")

######################################################################
# Add filter covariates 
######################################################################
df_X_covariates = pd.DataFrame({"training_instance_id": training_instance_id_list})
# Filter bank of (bandwidth, lag), in months: (1, 0), (3, 0), (6, 0), (12, 0), (12, 12), (12, 24). As in https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-014-0425-8
flat_filter_1 = Filter("flat", 30, 0)
flat_filter_2 = Filter("flat", 91, 0)
flat_filter_3 = Filter("flat", 182, 0)
flat_filter_4 = Filter("flat", 365, 0)
flat_filter_5 = Filter("flat", 365, 365)
flat_filter_6 = Filter("flat", 365, 2*365)
flat_filter_bank = [flat_filter_1, flat_filter_2, flat_filter_3, flat_filter_4, flat_filter_5, flat_filter_6]

gauss_filter_1 = Filter("gauss", 30, 0)
gauss_filter_2 = Filter("gauss", 91, 0)
gauss_filter_3 = Filter("gauss", 182, 0)
gauss_filter_4 = Filter("gauss", 365, 0)
gauss_filter_5 = Filter("gauss", 365, 365)
gauss_filter_6 = Filter("gauss", 365, 2*365)
gauss_filter_bank = [gauss_filter_1, gauss_filter_2, gauss_filter_3, gauss_filter_4, gauss_filter_5, gauss_filter_6]
full_filter_bank = flat_filter_bank + gauss_filter_bank

# Loop over training cases and add row values to df_X_covariates
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    period_start = value[1]
    period_end = value[2]
    end_of_history = period_start # The time at which history ends and the treatment of interest begins 
    patient = COMMPASS_patient_dictionary[patient_name]
    
    # Drugs 
    drug_filter_values = [compute_drug_filter_values(this_filter, patient, end_of_history) for this_filter in flat_filter_bank]
    # Flatten the array to get a list of features for all drugs, all covariates
    drug_filter_values = np.ndarray.flatten(np.array(drug_filter_values))
    #print(drug_filter_values[drug_filter_values != 0])
    column_names = ["DF"+str(int(iii)) for iii in range(len(drug_filter_values))]
    df_X_covariates.loc[training_instance_id,column_names] = drug_filter_values

    # M protein
    Mprotein_filter_values = [compute_Mprotein_filter_values(this_filter, patient, end_of_history) for this_filter in full_filter_bank]
    # Flatten the array to get a list of features for all drugs, all covariates
    Mprotein_filter_values = np.ndarray.flatten(np.array(Mprotein_filter_values))
    #print(Mprotein_filter_values) #[Mprotein_filter_values != 0])
    column_names = ["MF"+str(int(iii)) for iii in range(len(Mprotein_filter_values))]
    df_X_covariates.loc[training_instance_id,column_names] = Mprotein_filter_values

#print(df_X_covariates.head(n=5))
extracted_features = extracted_features_tsfresh.join(df_X_covariates)
#print(extracted_features.head(n=5))

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

r2scores_train = []
r2scores_test = []
print("Training and testing")
for ii in range(50):
    X_full_train, X_full_test, y_train, y_test = train_test_split(df_X_covariates, Y_parameters, test_size=.4, random_state=ii+randomnumberr)
    random_forest_model.fit(X_full_train, y_train)
    y_pred = random_forest_model.predict(X_full_test)
    y_pred_train = random_forest_model.predict(X_full_train)
    r2scores_train.append(r2_score(y_train, y_pred_train))
    r2scores_test.append(r2_score(y_test, y_pred))

######################################################################
print("Average R2 score train:", np.mean(r2scores_train), "std:", np.std(r2scores_train))
print("Average R2 score test:", np.mean(r2scores_test), "std:", np.std(r2scores_test))

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
    plt.savefig("./diagnostics_train_"+name+"_estimate_compared_to_truth.png")
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
plt.savefig("./diagnostics_pi_R_estimate_compared_to_truth.png")
plt.show()
"""

plt.figure()
plt.scatter(range(len(compare_g_r_test)), compare_g_r_test, c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(range(len(compare_g_r_pred)), compare_g_r_pred, c="red", s=s, edgecolor="black", label="Prediction")
plt.xlabel("Sorted by true g_r")
plt.ylabel("g_r: Growth rate of resistant cells")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.savefig("diagnostics_g_r_estimate_compared_to_truth.png")
plt.show()

plt.figure()
plt.scatter(range(len(compare_g_s_test)), compare_g_s_test, c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(range(len(compare_g_s_pred)), compare_g_s_pred, c="red", s=s, edgecolor="black", label="Prediction")
plt.xlabel("Sorted by true g_s")
plt.ylabel("g_s: Growth rate of resistant cells")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.savefig("diagnostics_g_s_estimate_compared_to_truth.png")
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

