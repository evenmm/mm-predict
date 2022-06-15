# Take history regions and estimates
# Extract features and learn the mapping from features to drug response parameters
from utilities import *
from sklearn import preprocessing
from sklearn import utils
start_time = time.time()
warnings.simplefilter("ignore")

# Load period definitions
training_instance_dict = np.load("training_instance_dict.npy", allow_pickle=True).item()
training_instance_id_list = [key for key in training_instance_dict.keys()] 
df_mprotein_and_dates = pd.read_pickle("df_mprotein_and_dates.pkl")
df_drugs_and_dates = pd.read_pickle("df_drugs_and_dates.pkl")
#print(df_mprotein_and_dates.head(n=5))

# load Y (parameters)
#picklefile = open('Y_parameters', 'rb')
#Y_parameters = pickle.load(picklefile)
#Y_parameters = [elem.to_prediction_array_composite_g_s_and_K_1()[0] for elem in Y_parameters]
#Y_parameters = np.array(Y_parameters)
#plt.hist(Y_parameters[:,0]) # Half mixture params zero, half nonzero. Interesting! (Must address how sensitive sensitive are too)
#plt.show()
#picklefile.close()

picklefile = open('Y_increase_or_not', 'rb')
Y_increase_or_not = pickle.load(picklefile)
lab = preprocessing.LabelEncoder()
Y_increase_or_not = lab.fit_transform(Y_increase_or_not)
picklefile.close()

print("Number of intervals:", len(Y_increase_or_not))
print("Number of 0s:", len(Y_increase_or_not) - np.count_nonzero(Y_increase_or_not))
print("Number of 1s:", sum(Y_increase_or_not[Y_increase_or_not == 1]))
#print("Number of nans:", sum(np.isnan(Y_increase_or_not)))
#print("Number of other things:", sum([(elem not in [0,1]) for elem in Y_increase_or_not]) - sum(np.isnan(Y_increase_or_not)))

picklefile = open('COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

picklefile = open('treatment_id_to_drugs_dictionary_COMMPASS', 'rb')
treatment_id_to_drugs_dictionary_COMMPASS = pickle.load(picklefile)
picklefile.close()

######################################################################
# tsfresh feature extraction from numerical columns in the dataframes 
######################################################################
# training_instance_dict is used to index into M protein and drug data frames
# Loop over all patients, create M protein, drug and clinical covariate data frames prior to tsfresh extraction 

# M protein
# Subsets taken from df_mprotein_and_dates and labeled by training_instance_id for tsfresh feature extraction 
df_Mprotein_pre_tsfresh = pd.DataFrame(columns=["D_LAB_serum_m_protein", "VISITDY", "training_instance_id", 'D_LAB_serum_kappa', 'D_LAB_serum_lambda'])
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    period_start = value[1] # This is the end of history
    end_of_history = period_start # The time at which history ends and the treatment of interest begins 
    #period_end = value[2] # This is irrelevant as it happens in the future
    treatment_id = value[3]
    
    single_entry = df_mprotein_and_dates[(df_mprotein_and_dates["PUBLIC_ID"] == patient_name) & (df_mprotein_and_dates["VISITDY"] < end_of_history)]
    single_entry["training_instance_id"] = training_instance_id
    df_Mprotein_pre_tsfresh = pd.concat([df_Mprotein_pre_tsfresh, single_entry])

    # Add one zero observation 10 years before diagnosis if the id is missing from the M protein df
    if training_instance_id not in pd.unique(df_Mprotein_pre_tsfresh[['training_instance_id']].values.ravel('K')):
        dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'D_LAB_serum_m_protein':[0], 'D_LAB_serum_kappa':[0], 'D_LAB_serum_lambda':[0], 'VISITDY':[-3650]})
        df_Mprotein_pre_tsfresh = pd.concat([df_Mprotein_pre_tsfresh, dummy], ignore_index=True)
#print(df_Mprotein_pre_tsfresh.head(n=20))

# Drugs
df_drugs_pre_tsfresh = pd.DataFrame(columns=['PUBLIC_ID', 'MMTX_THERAPY', 'drug_id', 'startday', 'stopday', "training_instance_id"])
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    period_start = value[1] # This is the end of history
    end_of_history = period_start # The time at which history ends and the treatment of interest begins 
    #period_end = value[2] # This is irrelevant as it happens in the future
    treatment_id = value[3]

    single_entry = df_drugs_and_dates[(df_drugs_and_dates["PUBLIC_ID"] == patient_name) & (df_drugs_and_dates["stopday"] <= end_of_history)]
    single_entry["training_instance_id"] = training_instance_id
    df_drugs_pre_tsfresh = pd.concat([df_drugs_pre_tsfresh, single_entry])
    
    # Add zero row if the id is missing from the drug df. This should have no effect at all since it has zero duration
    if training_instance_id not in pd.unique(df_drugs_pre_tsfresh[['training_instance_id']].values.ravel('K')):
        dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'drug_id':[0], 'startday':[end_of_history-3650], 'stopday':[end_of_history-3650]})
        df_drugs_pre_tsfresh = pd.concat([df_drugs_pre_tsfresh, dummy], ignore_index=True)
#print(df_drugs_pre_tsfresh.head(n=20))

# Feature extraction requires only numerical values in columns
from tsfresh import extract_features
df_Mprotein_pre_tsfresh = df_Mprotein_pre_tsfresh[["D_LAB_serum_m_protein", 'D_LAB_serum_kappa', 'D_LAB_serum_lambda', "VISITDY", "training_instance_id"]] # Remove PUBLIC_ID
#values = {"D_LAB_serum_m_protein":0, 'D_LAB_serum_kappa':0, 'D_LAB_serum_lambda':0} #"VISITDY":0, "training_instance_id":0
#df_Mprotein_pre_tsfresh = df_Mprotein_pre_tsfresh.fillna(value=values)
extracted_features_M_protein = extract_features(df_Mprotein_pre_tsfresh, column_id="training_instance_id", column_sort="VISITDY")

df_drugs_pre_tsfresh = df_drugs_pre_tsfresh[['drug_id','startday', 'stopday', "training_instance_id"]] # Remove PUBLIC_ID
extracted_features_drugs = extract_features(df_drugs_pre_tsfresh, column_id="training_instance_id", column_sort="startday")
###################################################################

# Impute nan and inf
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features_M_protein)
impute(extracted_features_drugs)

# Merge drug and M protein data 
extracted_features_tsfresh = extracted_features_drugs.join(extracted_features_M_protein) #, how="outer")
#print(extracted_features.head(n=5))

######################################################################
# Add filter covariates 
######################################################################
print("Adding filter covariates")
df_filter_covariates = pd.DataFrame({"training_instance_id": training_instance_id_list})
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

def add_filter_values(value_type, full_filter_bank, patient, end_of_history, df_filter_covariates, training_instance_id):
    filter_values = [compute_filter_values(this_filter, patient, end_of_history, value_type=value_type) for this_filter in full_filter_bank]
    # Flatten the array to get a list of features for all drugs, all covariates
    filter_values = np.ndarray.flatten(np.array(filter_values))
    column_names = [value_type+"F"+str(int(iii)) for iii in range(len(filter_values))]
    df_filter_covariates.loc[training_instance_id,column_names] = filter_values
    return df_filter_covariates

# Loop over training cases and add row values to df_filter_covariates
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    period_start = value[1] # This is the end of history
    end_of_history = period_start # The time at which history ends and the treatment of interest begins 
    #period_end = value[2] # This is irrelevant as it happens in the future
    treatment_id = value[3]
    
    # Drugs 
    drug_filter_values = [compute_drug_filter_values(this_filter, patient, end_of_history) for this_filter in flat_filter_bank]
    # Flatten the array to get a list of features for all drugs, all covariates
    drug_filter_values = np.ndarray.flatten(np.array(drug_filter_values))
    #print(drug_filter_values[drug_filter_values != 0])
    column_names = ["DF"+str(int(iii)) for iii in range(len(drug_filter_values))]
    df_filter_covariates.loc[training_instance_id,column_names] = drug_filter_values

    # M protein
    add_filter_values("Mprotein", full_filter_bank, patient, end_of_history, df_filter_covariates, training_instance_id)
    # Kappa 
    add_filter_values("Kappa", full_filter_bank, patient, end_of_history, df_filter_covariates, training_instance_id)
    # Lambda
    add_filter_values("Lambda", full_filter_bank, patient, end_of_history, df_filter_covariates, training_instance_id)

######################################################################
# Add clinical covariates and treatment as covariate
######################################################################
print("Adding clinical covariates and treatment as covariate")
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_V2.tsv'
df = pd.read_csv(filename, sep='\t')
df_IA17 = df[['PUBLIC_ID', 'ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender']]

df_clinical_covariates = pd.DataFrame(columns=["training_instance_id", 'PUBLIC_ID', 'ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender', '01Len', '01Dex', '01Bor', '01Melph', '01Cyclo'])
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    period_start = value[1] # This is the end of history
    end_of_history = period_start # The time at which history ends and the treatment of interest begins 
    #period_end = value[2] # This is irrelevant as it happens in the future
    treatment_id = value[3]
    
    single_entry = df_IA17[df_IA17["PUBLIC_ID"] == patient_name]
    single_entry["training_instance_id"] = training_instance_id

    # Future treatment 
    drug_names = get_drug_names_from_treatment_id_COMMPASS(treatment_id, treatment_id_to_drugs_dictionary_COMMPASS)
    single_entry['01Len'] = int('Lenalidomide' in drug_names)
    single_entry['01Dex'] = int('Dexamethasone' in drug_names)
    single_entry['01Bor'] = int('Bortezomib' in drug_names)
    single_entry['01Melph'] = int('Melphalan' in drug_names)
    single_entry['01Cyclo'] = int('Cyclophosphamide' in drug_names)

    df_clinical_covariates = pd.concat([df_clinical_covariates, single_entry])
# Sort by training_instance_id, drop that and PUBLIC_ID. Fillna values
df_clinical_covariates = df_clinical_covariates.sort_values(by=['training_instance_id'])
df_clinical_covariates.reset_index(drop=True, inplace=True)
df_clinical_covariates = df_clinical_covariates[['ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender', '01Len', '01Dex', '01Bor', '01Melph', '01Cyclo']]
values = {'ecog':0, 'DEMOG_PATIENTAGE':0, 'DEMOG_HEIGHT':0, 'DEMOG_WEIGHT':0, 'D_PT_race':1.1, 'D_PT_ethnic':1.1, 'D_PT_gender':1.5}
df_clinical_covariates = df_clinical_covariates.fillna(value=values)

dummy_df = pd.DataFrame(columns=["training_instance_id", 'PUBLIC_ID', 'ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender', '01Len', '01Dex', '01Bor', '01Melph', '01Cyclo'])
for training_instance_id, value in training_instance_dict.items():
    patient_name = value[0]
    patient = COMMPASS_patient_dictionary[patient_name]
    
    single_entry = df_IA17[df_IA17["PUBLIC_ID"] == patient_name]
    single_entry["training_instance_id"] = training_instance_id

    single_entry['01Len'] = int(np.random.randint(1))
    dummy_df = pd.concat([dummy_df, single_entry])
# Sort by training_instance_id, drop that and PUBLIC_ID. Fillna values
dummy_df = dummy_df.sort_values(by=['training_instance_id'])
dummy_df.reset_index(drop=True, inplace=True)
dummy_df = dummy_df[['01Len']]

# Three different types of covariates:
# extracted_features_tsfresh
# df_filter_covariates
# df_clinical_covariates
df_X_covariates = extracted_features_tsfresh.join(df_filter_covariates)
df_X_covariates = df_X_covariates.join(df_clinical_covariates)
#df_X_covariates = df_X_covariates[["DEMOG_PATIENTAGE"]]
print(df_X_covariates.head(n=5))

######################################################################
# Split into train and test 
######################################################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
# Train a random forest regressor
randomnumberr = 4219

r2score_train_array = []
r2score_test_array = []
print("Training and testing")

def train_random_forest(args):
    df_X_covariates, Y_outcome, i = args

    random_forest_model = RandomForestClassifier(n_estimators=1000, random_state=randomnumberr)
    X_train, X_test, y_train, y_test = train_test_split(df_X_covariates, Y_outcome, test_size=.4, stratify=Y_outcome, random_state=randomnumberr + i)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    y_pred_train = random_forest_model.predict(X_train)

    # calculate the fpr and tpr for all thresholds of the classification
    probs = random_forest_model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # train data too
    probs_train = random_forest_model.predict_proba(X_train)
    preds_train = probs_train[:,1]
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, preds_train)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)

    return (random_forest_model, X_train, X_test, y_train, y_test, y_pred, y_pred_train, probs, preds, fpr, tpr, roc_auc, probs_train, preds_train, fpr_train, tpr_train, threshold_train, roc_auc_train)

all_random_states = range(50)
args = [(df_X_covariates, Y_increase_or_not, i) for i in all_random_states]
with Pool(15) as pool:
    random_forest_model_array, X_train_array, X_test_array, y_train_array, y_test_array, y_pred_array, y_pred_train_array, probs_array, preds_array, fpr_array, tpr_array, roc_auc_array, probs_train_array, preds_train_array, fpr_train_array, tpr_train_array, threshold_train_array, roc_auc_train_array = zip(*pool.map(train_random_forest, args))
    #results = pool.map(train_random_forest, args)

#model_array = [elem[0] for elem in results]
#X_train_array = [elem[1] for elem in results]
#X_test_array = [elem[2] for elem in results]
#y_train_array = [elem[3] for elem in results]
#y_test_array = [elem[4] for elem in results]
#y_pred_array = [elem[5] for elem in results]
#y_pred_train_array = [elem[6] for elem in results]
#roc_auc_array = 

#model = model_array[-1]
#X_train = X_train_array[-1]
#X_test = X_test_array[-1]
#y_test = y_test_array[-1]
#y_train = y_train_array[-1]
#y_pred = y_pred_array[-1]
#y_pred_train = y_pred_train_array[-1]

random_forest_model = random_forest_model_array[-1]
X_train = X_train_array[-1]
X_test = X_test_array[-1]
y_train = y_train_array[-1]
y_test = y_test_array[-1]
y_pred = y_pred_array[-1]
y_pred_train = y_pred_train_array[-1]
probs = probs_array[-1]
preds = preds_array[-1]
fpr = fpr_array[-1]
tpr = tpr_array[-1]
roc_auc = roc_auc_array[-1]
probs_train = probs_train_array[-1]
preds_train = preds_train_array[-1]
fpr_train = fpr_train_array[-1]
tpr_train = tpr_train_array[-1]
threshold_train = threshold_train_array[-1]
roc_auc_train = roc_auc_train_array[-1]

end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)
print("Average AUC score on train data:", np.mean(roc_auc_train_array), "std:", np.std(roc_auc_train_array))
print("Average AUC score on test data:", np.mean(roc_auc_array), "std:", np.std(roc_auc_array))

# method I: plt
plt.title('Receiver Operating Characteristic, test data')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(visible=True)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("./ROC_test_data.png")
#plt.show()

print(classification_report(y_test, y_pred))

picklefile = open('random_forest_model', 'wb')
pickle.dump(random_forest_model, picklefile)
picklefile.close()

picklefile = open('df_X_covariates', 'wb')
pickle.dump(df_X_covariates, picklefile)
picklefile.close()

picklefile = open('X_test', 'wb')
pickle.dump(X_test, picklefile)
picklefile.close()

picklefile = open('y_test', 'wb')
pickle.dump(y_test, picklefile)
picklefile.close()
