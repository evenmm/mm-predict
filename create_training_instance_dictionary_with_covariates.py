from utilities import *
warnings.simplefilter("ignore")
import copy

def create_training_instance_dictionary_with_covariates(minimum_number_of_measurements=3, global_treatment_id_list=[1], threshold_for_closeness_for_M_protein_at_start=60, INCLUDE_SUBSEQUENT_DRUG_HOLIDAY = False, verbose=False):
    #minimum_number_of_measurements = 3 # required number of M protein measurements for a period to be included in the dataset
    #global_treatment_id_list = [1,2,3,7,10,13,15,16] #[38,62,66,110] are other combinations with the same drugs (but 66 only has 6 patients). # 30 for patient 1727 # Subset of treatment ids we choose to include. [15,16,3,10,7,1,13,2]: #range(1,unique_treat_counter):
    #threshold_for_closeness_for_M_protein_at_start = 60 # If M protein at period start is missing, it is imputed using the nearest measurement, but only if closer than this threshold number of days.
    #INCLUDE_SUBSEQUENT_DRUG_HOLIDAY = False #True # If a treatment is followed by a drug holiday, this decided if the holiday is included as part of the period

    # Load COMMPASS_patient_dictionary
    picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'rb')
    COMMPASS_patient_dictionary = pickle.load(picklefile)
    picklefile.close()

    picklefile = open('./binaries_and_pickles/unique_treat_counter', 'rb')
    unique_treat_counter = pickle.load(picklefile)
    picklefile.close()

    # Find the treatment id of the required treatment, extract those treatment regions and perform inference there
    def add_training_instance(training_instance_id, training_instance_dict, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, dummy_patient_dict, patient, treatment_id_of_interest):
        # Find the observed values, times and drug periods in the period
        valid_Mprotein = dummy_Mprotein_values[dummy_measurement_times>=period_start]
        valid_Kappa = dummy_Kappa_values[dummy_measurement_times>=period_start]
        valid_Lambda = dummy_Lambda_values[dummy_measurement_times>=period_start]
        valid_times = dummy_measurement_times[dummy_measurement_times>=period_start]
        valid_Mprotein = valid_Mprotein[valid_times<=period_end]
        valid_Kappa = valid_Kappa[valid_times<=period_end]
        valid_Lambda = valid_Lambda[valid_times<=period_end]
        valid_times = valid_times[valid_times<=period_end]
        # Check how many M protein values are within period, and only add as training instance if there are enough:
        if len(valid_times) >= minimum_number_of_measurements and max(valid_Mprotein) > 0 and valid_Mprotein[0] > 0.1:
            if verbose:
                print("Saving a case from", patient.name, "- treatment id", treatment_id_of_interest, "- training instance id", training_instance_id)
            how_many_regions[treatment_id_of_interest] = how_many_regions[treatment_id_of_interest] + 1
            # Note the time limits of this period: Last M protein measurement while still on treatment
            last_measurement_time_on_treatment = valid_times[-1]
            # Add covariates for this patient
            covariates = [period_start]
            # Create a dummy patient for this interval
            dummmy_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=this_history, Mprotein_values=valid_Mprotein, Kappa_values=valid_Kappa, Lambda_values=valid_Lambda, covariates=covariates, name="dummy")
            N_observations_this_period = len(dummmy_patient.measurement_times)
            
            training_instance_dict[training_instance_id] = [patient.name, period_start, period_end, treatment_id_of_interest, last_measurement_time_on_treatment]
            dummy_patient_dict[training_instance_id] = dummmy_patient
            training_instance_id = training_instance_id + 1


        return training_instance_id, training_instance_dict, how_many_regions, dummy_patient_dict, patient

    training_instance_id = 0
    training_instance_dict = {} # A dictionary mapping training_instance_id to the patient name and the start and end of the interval with the treatment of interest 
    dummy_patient_dict  = {} # A dictionary of dummy patients 

    # Iterate over all patients, look at their treatment periods one by one and check if it qualifies as a training item 
    start_time = time.time()
    how_many_regions = np.zeros(unique_treat_counter)
    for name, patient in COMMPASS_patient_dictionary.items():
        if len(patient.measurement_times) > minimum_number_of_measurements:
            treatment_id_list = copy.deepcopy(global_treatment_id_list)
            for outer_index, outer_treatment in enumerate(patient.treatment_history): # Outer loop so we pass each of them only once 
                if outer_treatment.id in treatment_id_list:
                    treatment_id_of_interest = outer_treatment.id

                    # New, simpler and faster version when not including treatment holidays
                    # Check if period satisfies, if so then add it as a training case
                    # *Just check the current period instead of looping, it is faster*
                    # *Then remove treatment_id_list popping at end

                    # Find periods of interest by looking through patient history 
                    period_start = np.nan
                    period_end = np.nan
                    valid_interval = False
                    dummy_measurement_times = np.array([])
                    dummy_Mprotein_values = np.array([])
                    dummy_Kappa_values = np.array([])
                    dummy_Lambda_values = np.array([])
                    this_history = np.array([])
                    for index, treatment in enumerate(patient.treatment_history[outer_index:]): # Inner loop to check what happens after this exact treatment from the outer loop
                        if valid_interval == False: 
                            if treatment.id == treatment_id_of_interest:
                                # We found the start 
                                valid_interval = True
                                period_start = treatment.start
                                this_history = np.array([treatment])
                                # Find the M protein value closest in time to the start of the treatment
                                distances_to_treatment_start = patient.measurement_times - np.repeat(treatment.start, len(patient.measurement_times))
                                abs_distances_to_treatment_start = abs(distances_to_treatment_start)
                                closest_index = np.argmin(abs_distances_to_treatment_start)
                                if (not abs_distances_to_treatment_start[closest_index] == 0) and (min(abs_distances_to_treatment_start) <= threshold_for_closeness_for_M_protein_at_start):
                                    # Add that value as M protein value at treatment start if there is nothing there
                                    if distances_to_treatment_start[closest_index] > 0: # Treatment.start comes before the value that was closest
                                        dummy_measurement_times = np.concatenate((patient.measurement_times[0:closest_index], [treatment.start], patient.measurement_times[closest_index:]))
                                        dummy_Mprotein_values = np.concatenate((patient.Mprotein_values[0:closest_index], [patient.Mprotein_values[closest_index]], patient.Mprotein_values[closest_index:]))
                                        dummy_Kappa_values = np.concatenate((patient.Kappa_values[0:closest_index], [patient.Kappa_values[closest_index]], patient.Kappa_values[closest_index:]))
                                        dummy_Lambda_values = np.concatenate((patient.Lambda_values[0:closest_index], [patient.Lambda_values[closest_index]], patient.Lambda_values[closest_index:]))
                                    else: # Treatment.start comes after the value that was closest. Indexing is ok, even in empty space
                                        dummy_measurement_times = np.concatenate((patient.measurement_times[0:closest_index+1], [treatment.start], patient.measurement_times[closest_index+1:]))
                                        dummy_Mprotein_values = np.concatenate((patient.Mprotein_values[0:closest_index+1], [patient.Mprotein_values[closest_index]], patient.Mprotein_values[closest_index+1:]))
                                        dummy_Kappa_values = np.concatenate((patient.Kappa_values[0:closest_index+1], [patient.Kappa_values[closest_index]], patient.Kappa_values[closest_index+1:]))
                                        dummy_Lambda_values = np.concatenate((patient.Lambda_values[0:closest_index+1], [patient.Lambda_values[closest_index]], patient.Lambda_values[closest_index+1:]))
                            #elif treatment.id == 0:
                            #    # The last M protein under this treatment might be the start
                            #    # Find time of last M protein value within period
                            #    valid_Mprotein = patient.Mprotein_values[patient.measurement_times>=treatment.start]
                            #    valid_times = patient.measurement_times[patient.measurement_times>=treatment.start]
                            #    valid_Mprotein = valid_Mprotein[valid_times<=treatment.end]
                            #    valid_times = valid_times[valid_times<=treatment.end]
                            #    if len(valid_times) > 0:
                            #        valid_interval = True
                            #        period_start = valid_times[-1]
                            #        zero_treatment = Treatment(period_start, treatment.end, treatment.id)
                            #        this_history = np.array([zero_treatment])
                            #        # Continute to next
                            else:
                                # Continue looking for the start
                                pass 
                        else: # if valid interval == True, then we are looking for 0 or the end 
                            if treatment.id == treatment_id_of_interest:
                                this_history = np.append(this_history,[treatment])
                                pass
                            elif treatment.id == 0 and INCLUDE_SUBSEQUENT_DRUG_HOLIDAY == True:
                                # Only after the id of interest. We are extending the period by a drug holiday, then ending the period at the end of this treatment
                                if patient.name == "MMRF_1793" and treatment_id_of_interest == 15:
                                    treatment = Treatment(treatment.start, 1770, treatment.id)
                                this_history = np.append(this_history,[treatment])
                                valid_interval = False
                                period_end = treatment.end
                                # Estimate parameters, add to Y and to the patient using the common function
                                training_instance_id, training_instance_dict, how_many_regions, dummy_patient_dict, patient = add_training_instance(training_instance_id, training_instance_dict, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, dummy_patient_dict, patient, treatment_id_of_interest)
                                dummy_measurement_times = np.array([])
                                dummy_Mprotein_values = np.array([])
                                dummy_Kappa_values = np.array([])
                                dummy_Lambda_values = np.array([])
                            else: 
                                # We found the end at the beginning of this foreign treatment id
                                valid_interval = False
                                # Check if we captured the treatment of interest and not only zero: 
                                if treatment_id_of_interest in [element.id for element in this_history]:
                                    period_end = treatment.start
                                    # Estimate parameters, add to Y and to the patient using the common function
                                    training_instance_id, training_instance_dict, how_many_regions, dummy_patient_dict, patient = add_training_instance(training_instance_id, training_instance_dict, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, dummy_patient_dict, patient, treatment_id_of_interest)
                                    dummy_measurement_times = np.array([])
                                    dummy_Mprotein_values = np.array([])
                                    dummy_Kappa_values = np.array([])
                                    dummy_Lambda_values = np.array([])
                    # After the last treatment, if ending on a valid interval, this is the end
                    if valid_interval == True: 
                        # Check if we captured the treatment of interest and not only zero: 
                        if treatment_id_of_interest in [element.id for element in this_history]:
                            period_end = patient.treatment_history[-1].end
                            # Estimate parameters, add to Y and to the patient using the common function
                            training_instance_id, training_instance_dict, how_many_regions, dummy_patient_dict, patient = add_training_instance(training_instance_id, training_instance_dict, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, dummy_patient_dict, patient, treatment_id_of_interest)
                    treatment_id_list.remove(treatment_id_of_interest)


    # Covariates must take training_instance_dict as an argument
    ## Load covariate dataframe X
    #picklefile = open('./binaries_and_pickles/df_X_covariates', 'rb')
    #df_X_covariates = pickle.load(picklefile)
    #picklefile.close()

    end_time = time.time()
    time_duration = end_time - start_time
    if verbose:
        for ii in range(len(dummy_patient_dict)):
            dummy_patient_dict[ii].print()
    print("Done finding", len(training_instance_dict.values()), "training instances with at least", minimum_number_of_measurements, "M protein measurements.")
    np.save("./binaries_and_pickles/training_instance_dict.npy", training_instance_dict)
    np.save("./binaries_and_pickles/dummy_patient_dict.npy", dummy_patient_dict)
    return dummy_patient_dict, training_instance_dict

if __name__ == "__main__":
    create_training_instance_dictionary_with_covariates()    
