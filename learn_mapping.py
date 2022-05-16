# Take history regions and estimates
# Extract features and learn the mapping from features to drug response parameters
from utilities import *

# Load X (period definitions) and Y (parameters)
X_periods = np.load("X_periods.npy", allow_pickle=True).item()
df_mprotein_and_dates = pd.read_pickle("df_mprotein_and_dates.pkl")
df_drugs_and_dates = pd.read_pickle("df_drugs_and_dates.pkl")
#print(df_mprotein_and_dates.head(n=5))

picklefile = open('Y_parameters', 'rb')
Y_parameters = pickle.load(picklefile)
Y_parameters = [elem.to_prediction_array_composite_g_s_and_K_1() for elem in Y_parameters]
Y_parameters = np.array(Y_parameters)
#print(Y_parameters)
#plt.hist(Y_parameters[:,0]) # Half mixture params zero, half nonzero. Interesting! (Must address how sensitive sensitive are too)
#plt.show()

picklefile.close()
picklefile = open('COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

# Prepare data by choosing only int & float valued columns 
df_mprotein_and_dates = df_mprotein_and_dates[["training_instance_id", "D_LAB_serum_m_protein", "VISITDY"]]
patient_with_training_instance_id_57 = pd.DataFrame({"training_instance_id":[57], "D_LAB_serum_m_protein":[0], "VISITDY":[0]})
df_mprotein_and_dates = pd.concat([df_mprotein_and_dates, patient_with_training_instance_id_57], ignore_index=True)

training_instance_id_list = pd.unique(df_mprotein_and_dates[['training_instance_id']].values.ravel('K'))

# Prepare data by choosing only int & float valued columns: drug_id instead of MMTX_THERAPY
df_drugs_and_dates = df_drugs_and_dates[['drug_id','startday', 'stopday', "training_instance_id"]]
for training_instance_id in training_instance_id_list: 
    if training_instance_id not in pd.unique(df_drugs_and_dates[['training_instance_id']].values.ravel('K')):
        dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'drug_id':[0],'startday':[0], 'stopday':[0]})
        df_drugs_and_dates = pd.concat([df_drugs_and_dates, dummy], ignore_index=True)

from tsfresh import extract_features
extracted_features_M_protein = extract_features(df_mprotein_and_dates, column_id="training_instance_id", column_sort="VISITDY")

extracted_features_drugs = extract_features(df_drugs_and_dates, column_id="training_instance_id", column_sort="startday")

# Impute nan and inf
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features_M_protein)
impute(extracted_features_drugs)
print(extracted_features_M_protein.index)
print(extracted_features_drugs.index)

# Merge drug and M protein data 
extracted_features = extracted_features_drugs.join(extracted_features_M_protein) #, how="outer", on="training_instance_id")

randomnumberr = 4219
# Add the drugs: 
# ...

#print(extracted_features.index) #.head(n=5))
# Split into train and test 
from sklearn.model_selection import train_test_split
#print(extracted_features.loc[].head(n=5))
#for col in extracted_features.columns:
#    print(col)
#print(len(extracted_features))
#print(len(Y_parameters))
#print(Y_parameters)
X_full_train, X_full_test, y_train, y_test = train_test_split(extracted_features, Y_parameters, test_size=.4, random_state=randomnumberr)

#print("X_full_test")
#print(X_full_test.head(n=5))

# Train a very naive decision tree (Full)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
random_forest_model = RandomForestRegressor(random_state=randomnumberr)
random_forest_model.fit(X_full_train, y_train)
y_pred = random_forest_model.predict(X_full_test)
y_pred_train = random_forest_model.predict(X_full_train)
print("R2 score train:", r2_score(y_train, y_pred_train))
print("R2 score test:", r2_score(y_test, y_pred))

# Print true and estimated
#for index, elem in enumerate(y_pred):
#    print(y_test[index][0], ":", y_pred[index][0])
#    print(y_test[index][1], ":", y_pred[index][1])
#    print(y_test[index][2], ":", y_pred[index][2], "\n")

plt.figure()
s = 25
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

# These comparison lists of parameters contain pairs of (prediction, truth), sorted by truth, for plotting 
compare_pi_r = [[y_pred[ii][0], y_test[ii][0]] for ii, elem in enumerate(y_test)]
compare_g_r = [[y_pred[ii][1], y_test[ii][1]] for ii, elem in enumerate(y_test)]
compare_g_s = [[y_pred[ii][2], y_test[ii][2]] for ii, elem in enumerate(y_test)]

def sort_by_test(pred_in, test_in, index):
    # index 0: pi_r. index 1: g_r
    compare = [[pred_in[ii][index], test_in[ii][index]] for ii, elem in enumerate(test_in)]
    sorted_compare = Sort(compare)
    pred_array = [elem[0] for elem in sorted_compare]
    test_array = [elem[1] for elem in sorted_compare]
    return pred_array, test_array

compare_pi_r = Sort(compare_pi_r)
compare_pi_r_pred = [elem[0] for elem in compare_pi_r]
compare_pi_r_test = [elem[1] for elem in compare_pi_r]

compare_g_r = Sort(compare_g_r)
compare_g_r_pred = [elem[0] for elem in compare_g_r]
compare_g_r_test = [elem[1] for elem in compare_g_r]

compare_g_s = Sort(compare_g_s)
compare_g_s_pred = [elem[0] for elem in compare_g_s]
compare_g_s_test = [elem[1] for elem in compare_g_s]

def make_figure(pred_array, test_array, name):
    plt.figure()
    plt.scatter(range(len(pred_array)), test_array, c="navy", s=s, edgecolor="black", label="Data")
    plt.scatter(range(len(pred_array)), pred_array, c="red", s=s, edgecolor="black", label="Prediction")
    plt.xlabel("Sorted by true "+name)
    plt.ylabel(name+": Fraction resistant cells")
    plt.title("Train data: Compare truth and predictions")
    plt.legend(loc="best")
    plt.savefig("diagnostics_train_"+name+"_estimate_compared_to_truth.png")
    plt.show()

# pi_r, g_r, g_s
#for iii in range(3):
pred_array, test_array = sort_by_test(y_pred_train, y_train, 0)
make_figure(pred_array, test_array, "pi_r")
pred_array, test_array = sort_by_test(y_pred_train, y_train, 1)
make_figure(pred_array, test_array, "g_r")
pred_array, test_array = sort_by_test(y_pred_train, y_train, 2)
make_figure(pred_array, test_array, "g_s")

plt.figure()
plt.scatter(range(len(compare_pi_r_test)), compare_pi_r_test, c="navy", s=s, edgecolor="black", label="Data")
plt.scatter(range(len(compare_pi_r_pred)), compare_pi_r_pred, c="red", s=s, edgecolor="black", label="Prediction")
plt.xlabel("Sorted by true pi_R")
plt.ylabel("pi_R: Fraction resistant cells")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.savefig("diagnostics_pi_R_estimate_compared_to_truth.png")
plt.show()

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


