# mm-data
Description and treatment of longitudinal data of Multiple myeloma patients

Most scripts depend on utilities.py
utilities.py contains utility functions for inference 
It depends on plot_COMMPASS and plot_data_overview having run first, to create the drug dictionaries. 

First, make sure to download the necessary files from the COMMPASS dataset by the Multiple Myeloma Research Foundation
Then, run the following scripts in order: 
COMMPASS_patient_inference.py 
learn_mapping.py 
#evaluate_and_plot_results.py
