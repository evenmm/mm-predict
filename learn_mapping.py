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

df_drugs_and_dates = df_drugs_and_dates[['MMTX_THERAPY','startday', 'stopday', "training_instance_id"]]

from tsfresh import extract_features
extracted_features_M_protein = extract_features(df_mprotein_and_dates, column_id="training_instance_id", column_sort="VISITDY")

#extracted_features_drugs = extract_features(df_drugs_and_dates, column_id="training_instance_id", column_sort="startday")

# Impute nan and inf
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features_M_protein)
#impute(extracted_features_drugs)
print(extracted_features_M_protein.index)
#print(extracted_features_drugs.index)

# Merge drug and M protein data 


randomnumberr = 4219
# Add the drugs: 
# ...

#print(extracted_features_M_protein.index) #.head(n=5))
# Split into train and test 
from sklearn.model_selection import train_test_split
#print(extracted_features_M_protein.loc[].head(n=5))
#for col in extracted_features_M_protein.columns:
#    print(col)
#print(len(extracted_features_M_protein))
#print(len(Y_parameters))
#print(Y_parameters)
X_full_train, X_full_test, y_train, y_test = train_test_split(extracted_features_M_protein, Y_parameters, test_size=.4, random_state=randomnumberr)

#print("X_full_test")
#print(X_full_test.head(n=5))

# Train a very naive decision tree (Full)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
random_forest_model = RandomForestRegressor(random_state=randomnumberr)
random_forest_model.fit(X_full_train, y_train)
y_pred = random_forest_model.predict(X_full_test)
print("R2 score:", r2_score(y_test, y_pred))

# Print true and estimated
#for index, elem in enumerate(y_pred):
#    print(y_test[index][0], ":", y_pred[index][0])
#    print(y_test[index][1], ":", y_pred[index][1])
#    print(y_test[index][2], ":", y_pred[index][2], "\n")

plt.figure()
s = 25
plt.scatter(y_test[:, 0], y_test[:, 1], c="navy", s=s, edgecolor="black", label="data")
plt.scatter(y_pred[:, 0], y_pred[:, 1], c="red", s=s, edgecolor="black", label="Prediction")
#plt.xlim([-6, 6])
#plt.ylim([-6, 6])
plt.xlabel("pi_R: Fraction resistant cells")
plt.ylabel("g_r")
plt.title("Compare truth and predictions")
plt.legend(loc="best")
plt.show()

# Use latest M protein as Y_0 
#for training_instance_id in training_instance_id_list:
# iterate the patient names in X_full_test:
##for index, row in extracted_features_M_protein.iterrows():
#    training_instance_id = row['training_instance_id']
#    PUBLIC_ID = row['PUBLIC_ID']
#    patient = COMMPASS_patient_dictionary[PUBLIC_ID]
#
#    # Plot estimated compared to true... parameters? No, M protein values 
#    estimated_parameters = y[training_instance_id]
#    predicted_parameters = y_pred[training_instance_id]
#    savename = "./COMPASS_plot_comparisons/compare_predicted_with_estimated"+patient.name+".png"
#    plot_to_compare_estimated_and_predicted_drug_dynamics(estimated_parameters, predicted_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=patient.name, savename=savename)


