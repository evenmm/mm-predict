# Load patient data, find relevant sections, and perform inference 
from utilities import *
 
# M protein data
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_VISIT_V2.tsv'
df = pd.read_csv(filename, sep='\t')
df_mprotein_and_dates = df[['PUBLIC_ID', 'VISITDY',
'D_LAB_serum_m_protein',
#'D_IM_LIGHT_CHAIN_BY_FLOW', #"kappa" or "lambda"
#'D_LAB_serum_kappa', # Serum Kappa (mg/dL)
#'D_LAB_serum_lambda', # Serum Lambda (mg/dL)
#'D_IM_kaplam'
]]

# Here is a dictionary of patients indexed by their names: Name (str) --> Patient object (Patient)
COMMPASS_patient_dictionary = {}

# Iterate the M protein file and make patients of all the entries
for index, row in df_mprotein_and_dates.iterrows():
    patient_name = row['PUBLIC_ID']
    if not patient_name in COMMPASS_patient_dictionary.keys:
        COMMPASS_patient_dictionary[patient_name] = COMMPASS_Patient(measurement_times=[row['VISITDY']], drug_dates=set(), drug_history=[], treatment_history=[], observed_values=[row['D_LAB_serum_m_protein']], covariates=[], name=patient_name)
    else:
        COMMPASS_patient_dictionary[patient_name].add_Mprotein_line_to_patient(row['VISITDY'], row['D_LAB_serum_m_protein'])

# Add drugs 
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_STAND_ALONE_TREATMENT_REGIMEN_V2.tsv'
df = pd.read_csv(filename, sep='\t')
df_drugs_and_dates = df[[
    'PUBLIC_ID', 'MMTX_THERAPY', 
    'startday', 'stopday'
]]
for index, row in df_drugs_and_dates.iterrows():
    patient_name = row['PUBLIC_ID']
    drug_name = row['MMTX_THERAPY']
    drug_id = drug_dictionary[drug_name]
    drug_object = Drug_period(row['startday'], row['stopday'], drug_id)
    if not patient_name in COMMPASS_patient_dictionary.keys:
        COMMPASS_patient_dictionary[patient_name] = COMMPASS_Patient(measurement_times=[], drug_dates=set(), drug_history=[drug_object], treatment_history=[], observed_values=[], covariates=[], name=patient_name)
    else:
        COMMPASS_patient_dictionary[patient_name].add_drug_period_to_patient(drug_object)

# Generate treatment histories from drug histories for all patients:
# Create a set of dates wherever the treatment changed, including first and last M protein measurements if they are before or after first and last drug date
for patient_name, patient_object in COMMPASS_patient_dictionary:
    date_set = patient_object.drug_dates
    if min(patient_object.measurement_times) < min(date_set):
        date_set.add(min(patient_object.measurement_times))
    if max(patient_object.measurement_times) > max(date_set):
        date_set.add(max(patient_object.measurement_times))
    # Iterate through the date set and keep track of which drugs are being turned on and off 
    drug_set=[]
    for date in date_set:
        

    for drug_period in patient_object.drug_history:


# Load patient MMRF_1143 into a Patient object 
patient_MMRF_1143 = load_MMRF_patient_into_Patient_object()

