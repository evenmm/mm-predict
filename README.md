# mm-data
Description and treatment of longitudinal data of Multiple myeloma patients

Most scripts depend on utilities.py

utilities.py contains utility functions for inference 

It depends on plot_COMMPASS and plot_data_overview having run first, to create the drug dictionaries. 

First, make sure to download the necessary files from the COMMPASS dataset by the Multiple Myeloma Research Foundation

Then, run the following scripts in order: 

create_COMMPASS_patient_dictionary.py to get a dictionary of patients

COMMPASS_patient_inference.py to estimate drug response parameters Y; a dictionary of training instances containing 

    [patient.name, period_start, period_end, last_measurement_time_on_treatment, treatment_id_of_interest]

feature_extraction.py to extract features from the history of each training instance, save these in dataframe df_X_covariates

Either: 

    learn_mapping.py to predict all or a subset of parameters Y

    --or-- 

    binary_outcome.py to predict whether M protein increases or not after a certain time

feature_importance.py to determine which features are important
