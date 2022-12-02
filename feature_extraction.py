# Purposes of this script: 
#   Take predefined history regions (in training_instance_dict)
#   Perform feature extraction on M protein and drug history, using:
#   - tsfresh
#   - filters: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-014-0425-8
#   - Clinical covariates
#   - EHR covariates?
#   Save df_X_covariates

from utilities import *
start_time = time.time()
warnings.simplefilter("ignore")

def feature_extraction(training_instance_dict):
    training_instance_id_list = [key for key in training_instance_dict.keys()] 
    df_mprotein_and_dates = pd.read_pickle("./binaries_and_pickles/df_mprotein_and_dates.pkl")
    df_drugs_and_dates = pd.read_pickle("./binaries_and_pickles/df_drugs_and_dates.pkl")
    #print(df_mprotein_and_dates.head(n=5))

    picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'rb')
    COMMPASS_patient_dictionary = pickle.load(picklefile)
    picklefile.close()

    picklefile = open('./binaries_and_pickles/treatment_id_to_drugs_dictionary_COMMPASS', 'rb')
    treatment_id_to_drugs_dictionary_COMMPASS = pickle.load(picklefile)
    picklefile.close()

    """
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
        last_measurement_time_on_treatment = value[4]
        
        single_entry = df_mprotein_and_dates[(df_mprotein_and_dates["PUBLIC_ID"] == patient_name) & (df_mprotein_and_dates["VISITDY"] < end_of_history)]
        single_entry["training_instance_id"] = training_instance_id
        df_Mprotein_pre_tsfresh = pd.concat([df_Mprotein_pre_tsfresh, single_entry])

        # Add one zero observation 10 years before diagnosis if the id is missing from the M protein df
        if training_instance_id not in pd.unique(df_Mprotein_pre_tsfresh[['training_instance_id']].values.ravel('K')):
            dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'D_LAB_serum_m_protein':[0], 'D_LAB_serum_kappa':[0], 'D_LAB_serum_lambda':[0], 'VISITDY':[-3650]})
            df_Mprotein_pre_tsfresh = pd.concat([df_Mprotein_pre_tsfresh, dummy], ignore_index=True)
    #print(df_Mprotein_pre_tsfresh.head(n=20))
    # Feature extraction requires only numerical values in columns
    df_Mprotein_pre_tsfresh = df_Mprotein_pre_tsfresh[["D_LAB_serum_m_protein", 'D_LAB_serum_kappa', 'D_LAB_serum_lambda', "VISITDY", "training_instance_id"]] # Remove PUBLIC_ID

    # Drugs
    df_drugs_pre_tsfresh = pd.DataFrame(columns=['PUBLIC_ID', 'MMTX_THERAPY', 'drug_id', 'startday', 'stopday', "training_instance_id"])
    for training_instance_id, value in training_instance_dict.items():
        patient_name = value[0]
        patient = COMMPASS_patient_dictionary[patient_name]
        period_start = value[1] # This is the end of history
        end_of_history = period_start # The time at which history ends and the treatment of interest begins 
        #period_end = value[2] # This is irrelevant as it happens in the future
        treatment_id = value[3]
        last_measurement_time_on_treatment = value[4]

        single_entry = df_drugs_and_dates[(df_drugs_and_dates["PUBLIC_ID"] == patient_name) & (df_drugs_and_dates["stopday"] <= end_of_history)]
        single_entry["training_instance_id"] = training_instance_id
        df_drugs_pre_tsfresh = pd.concat([df_drugs_pre_tsfresh, single_entry])
        
        # Add zero row if the id is missing from the drug df. This should have no effect at all since it has zero duration
        if training_instance_id not in pd.unique(df_drugs_pre_tsfresh[['training_instance_id']].values.ravel('K')):
            #dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'drug_id':[0], 'startday':[end_of_history-3650], 'stopday':[end_of_history-3650]})
            dummy = pd.DataFrame({"training_instance_id":[training_instance_id], 'drug_id':[0], 'startday':[0], 'stopday':[0]})
            df_drugs_pre_tsfresh = pd.concat([df_drugs_pre_tsfresh, dummy], ignore_index=True)
    df_drugs_pre_tsfresh["duration"] = df_drugs_pre_tsfresh["stopday"] - df_drugs_pre_tsfresh["startday"]
    #df_drugs_pre_tsfresh = df_drugs_pre_tsfresh[['drug_id','startday', 'stopday', "training_instance_id"]] # Remove PUBLIC_ID and duration
    df_drugs_pre_tsfresh = df_drugs_pre_tsfresh[['drug_id','startday', 'duration', "training_instance_id"]] # Remove PUBLIC_ID and stopday
    #print(df_drugs_pre_tsfresh.isna().any())
    #print(df_drugs_pre_tsfresh.columns[df_drugs_pre_tsfresh.isna().any()].tolist())

    from tsfresh import extract_features
    print("M protein time series:")
    #values = {"D_LAB_serum_m_protein":0, 'D_LAB_serum_kappa':0, 'D_LAB_serum_lambda':0} #"VISITDY":0
    #df_Mprotein_pre_tsfresh = df_Mprotein_pre_tsfresh.fillna(value=values)
    extracted_features_M_protein = extract_features(df_Mprotein_pre_tsfresh, column_id="training_instance_id", column_sort="VISITDY")

    values = {'drug_id':0,'startday':0, 'duration':0, "training_instance_id":0} #"VISITDY":0
    df_drugs_pre_tsfresh = df_drugs_pre_tsfresh.fillna(value=values)
    print("Drug start and stop day time series:")
    extracted_features_drugs = extract_features(df_drugs_pre_tsfresh, column_id="training_instance_id", column_sort="startday")
    ###################################################################

    # Impute nan and inf
    from tsfresh.utilities.dataframe_functions import impute
    impute(extracted_features_M_protein)
    impute(extracted_features_drugs)

    # Merge drug and M protein data 
    extracted_features_tsfresh = extracted_features_drugs.join(extracted_features_M_protein) #, how="outer")
    print("Number of tsfresh covariates:", len(extracted_features_tsfresh.columns))
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
        last_measurement_time_on_treatment = value[4]

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

    df_filter_covariates.drop("training_instance_id", axis=1, inplace=True)
    print("Number of filter covariates:", len(df_filter_covariates.columns))
    """
    ######################################################################
    # Add clinical covariates and treatment as covariate
    ######################################################################
    print("Adding clinical covariates and treatment as covariate")
    filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_V2.tsv'
    df = pd.read_csv(filename, sep='\t')
    df_IA17 = df[['PUBLIC_ID', 'ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender']]

    df_clinical_covariates = pd.DataFrame(columns=["training_instance_id", 'PUBLIC_ID', 'ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender', '01Len', '01Dex', '01Bor', '01Melph', '01Cyclo', 'time_since_diagnosis'])
    for training_instance_id, value in training_instance_dict.items():
        patient_name = value[0]
        patient = COMMPASS_patient_dictionary[patient_name]
        period_start = value[1] # This is the end of history
        end_of_history = period_start # The time at which history ends and the treatment of interest begins 
        #period_end = value[2] # This is irrelevant as it happens in the future
        treatment_id = value[3]
        last_measurement_time_on_treatment = value[4]
        
        single_entry = df_IA17[df_IA17["PUBLIC_ID"] == patient_name]
        single_entry["training_instance_id"] = training_instance_id
        single_entry["time_since_diagnosis"] = end_of_history

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
    # Int from object to int
    df_clinical_covariates['01Len'] = df_clinical_covariates['01Len'].astype(int)
    df_clinical_covariates['01Dex'] = df_clinical_covariates['01Dex'].astype(int)
    df_clinical_covariates['01Bor'] = df_clinical_covariates['01Bor'].astype(int)
    df_clinical_covariates['01Melph'] = df_clinical_covariates['01Melph'].astype(int)
    df_clinical_covariates['01Cyclo'] = df_clinical_covariates['01Cyclo'].astype(int)
    df_clinical_covariates = df_clinical_covariates[['ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender', '01Len', '01Dex', '01Bor', '01Melph', '01Cyclo', 'time_since_diagnosis']]
    values = {'ecog':0, 'DEMOG_PATIENTAGE':0, 'DEMOG_HEIGHT':0, 'DEMOG_WEIGHT':0, 'D_PT_race':1.1, 'D_PT_ethnic':1.1, 'D_PT_gender':1.5}
    df_clinical_covariates = df_clinical_covariates.fillna(value=values)
    print("Number of clinical covariates including drug indicators:", len(df_clinical_covariates.columns))

    ######################################################################
    # Add EHR clinical covariates
    ######################################################################
    print("Adding clinical covariates by EHR")
    filename = './COMMPASS_data/220615_commpass_clinical_genomic_annotated_EHR.xlsx'
    df_EHR = pd.read_excel(filename)
    # Imputation of missing cases must be done after merging to see which patients were not included here
    COMMPASS_current_name_list = [elem[0] for elem in training_instance_dict.values()]
    EHR_name_list = [elem.replace("_1_BM" ,"", 1) for elem in df_EHR.loc[:,"sample"]]
    #print("\nCOMMPASS_current_name_list:\n", len(COMMPASS_current_name_list))
    
    #print("\nEHR_name_list:\n", len(EHR_name_list))
    #print([(elem, (elem not in COMMPASS_current_name_list)) for elem in df_EHR.loc[:,"sample"]])
    #print([(elem not in COMMPASS_current_name_list) for elem in EHR_name_list])
    print("How many from EHR not in current COMMPASS name list (due to inclusion criteria not being satisfied)\n:", sum([(elem not in COMMPASS_current_name_list) for elem in EHR_name_list]))
    print("How many from current COMMPASS not in EHR name list:", sum([(elem not in EHR_name_list) for elem in COMMPASS_current_name_list]))
    print("How many overlapping:", sum([(elem in EHR_name_list) for elem in COMMPASS_current_name_list]))

    # Sanitize data types by excluding Age and Gender which were already included, and removing "sample", which is name
    # Age is a categorical variable: '<=70' or '>70'
    # D_PT_gender is 'Male'/'Female'
    # 'sample' is a string, the patient name
    df_EHR.drop(columns=["sample","D_PT_gender","Age"], inplace=True)

    # Make sure we actually join on the patient name. The order is the same as it is.

    ######################################################################
    # Save things
    ######################################################################
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
    #df_X_covariates = extracted_features_tsfresh.join(df_filter_covariates)
    #df_X_covariates = df_X_covariates.join(df_clinical_covariates)
    df_X_covariates = df_clinical_covariates
    #df_X_covariates = df_clinical_covariates
    #df_X_covariates = df_X_covariates[["DEMOG_PATIENTAGE"]]

    # Removing the homebrewed ones: 
    print("Removing the homebrewed ones...")
    df_X_covariates.drop(columns=['ecog', 'DEMOG_PATIENTAGE', 'DEMOG_HEIGHT', 'DEMOG_WEIGHT', 'D_PT_race', 'D_PT_ethnic', 'D_PT_gender', '01Len', '01Dex', '01Bor', '01Melph', '01Cyclo', 'time_since_diagnosis'], inplace=True)
    # EHR:
    df_X_covariates = df_X_covariates.join(df_EHR, how='left', lsuffix='_left', rsuffix='_right') # "D_PT_gender" was present in both: This preserves both, suffixed by _left/_right
    ## This check reveals that they are sorted in the same way: 
    #df_X_covariates = df_X_covariates[["D_PT_gender_left", "D_PT_gender_left"]]
    #print(df_X_covariates.head(n=50))
    
    #df_X_covariates = extracted_features_tsfresh.join(df_EHR, how='left', lsuffix='_left', rsuffix='_right')
    #df_X_covariates = df_X_covariates.join(df_clinical_covariates)
    #df_X_covariates = df_X_covariates.join(df_EHR)
    #df_X_covariates = df_clinical_covariates.join(df_EHR, how='left', lsuffix='_left', rsuffix='_right')
    #print("Impute for missing values in EHR's data until we can recompute them all")
    #df_X_covariates.fillna(df_X_covariates.mean())
    #print(df_X_covariates.isna().any())
    #print(df_X_covariates.columns[df_X_covariates.isna().any()].tolist())

    ######################################################################
    # Data cleaning
    ######################################################################
    print("Any nan values in df_EHR?",df_EHR.isnull().values.any())
    print("Any nan values in df_clinical_covariates?",df_clinical_covariates.isnull().values.any())
    answer_to_question = df_X_covariates.isnull().values.any()
    print("Any nan values in df_X_covariates?", answer_to_question)
    if answer_to_question == True:
        print("Which are nan values in df_EHR?\n",df_X_covariates.isnull())
    
    print("These columns have nan values:", df_X_covariates.columns[df_X_covariates.isna().any()].tolist())
    for colname in df_X_covariates.columns:
        the_nan_sum = df_X_covariates[colname].isnull().sum()
        if the_nan_sum > 0:
            print(colname, "has", the_nan_sum, "nan values.")
            mean_value = df_X_covariates[colname].mean()
            df_X_covariates[colname].fillna(value=mean_value, inplace=True)
    print("Missing data filled with naive means not per group")
    print("These columns have nan values:", df_X_covariates.columns[df_X_covariates.isna().any()].tolist())

    # Standardization: 
    print("Before standardizing:")
    #hist = df_X_covariates.hist(column="D_PT_age")
    #plt.show()
    #hist = df_X_covariates.hist(column="oscdy")
    #plt.show()
    #hist = df_X_covariates.hist(column="pfscdy")
    #plt.show()

    print(df_X_covariates.head(n=10))
    print("Standardizing some...")
    for colname in ["D_PT_age", "oscdy", "pfscdy"]: 
        df_X_covariates[colname] = ( df_X_covariates[colname] - df_X_covariates[colname].mean() ) / df_X_covariates[colname].std()
    #print("Normalizing some...")
    #for colname in ["pfscdy"]: # Putting them between -1 and 1
    #    df_X_covariates[colname] = -0.5 + ( df_X_covariates[colname] - df_X_covariates[colname].min() ) / ( df_X_covariates[colname].max() - df_X_covariates[colname].min() )

    print("Total number of covariates in df_X:", len(df_X_covariates.columns))
    print(df_X_covariates.head(n=10))

    # Categorical variables that are ordered and don't need one-hot encoding: 
    # ecog = Eastern Cooperative Oncology Group Performance Status Scale
    # ISS = International Staging System (ISS) for Multiple Myeloma
    #"D_PT_iss", "ecog", "ISS"

    # This prints all the data types: 
    #df_X_covariates.info(verbose=True)

    # Save dataframe X with covairates
    picklefile = open('./binaries_and_pickles/df_X_covariates', 'wb')
    pickle.dump(df_X_covariates, picklefile)
    picklefile.close()
    return df_X_covariates

if __name__ == "__main__":
    # Load period definitions
    training_instance_dict = np.load("./binaries_and_pickles/training_instance_dict.npy", allow_pickle=True).item()
    df_X_covariates = feature_extraction(training_instance_dict)
