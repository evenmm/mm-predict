nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
next_treatment_lines_this_patient = 1
for row_index in range(len(df_mprotein_and_dates)):
    # Check if date and treatment exists. 
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    missing_date_bool = isNaN(drug_interval_1).any()
    # Check if radiation or missing date
    if (not missing_date_bool) and (not (df_mprotein_and_dates.loc[row_index,['Drug 1']][0] == "Radiation")):
        drugs_1 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']])
        drugs_1 = drugs_1[~isNaN(drugs_1)]
        this_drug_set = []
        for ii in range(len(drugs_1)):
            this_drug_set.append(drugs_1[ii]) 
        this_treatment_line = frozenset(this_drug_set)
        # If new patient
        if not (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
            nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
            next_treatment_lines_this_patient = 1
            df_treatment_lines.loc[len(df.index)] = [nnid, this_treatment_line] + np.repeat(np.nan, 19).tolist
            next_treatment_lines_this_patient = next_treatment_lines_this_patient + 1
        else:
            # Add treatment line to existing patient
            df_mprotein_and_dates.loc[row_index, "Treatment line "+str(next_treatment_lines_this_patient)] = this_treatment_line
            next_treatment_lines_this_patient = next_treatment_lines_this_patient + 1
            

#foo = [55.97851164, 35.73303812, 48.15100092, np.nan, np.nan]
#i = np.where(np.isnan(foo))[0][0] - 1
#print(i)

#dumdumdict = {99:"99", 100:"100", 200:"200"}
#for ii in range(20):
#    dumdumdict[ii] = ii
#
#for ii, mtimes in enumerate(dumdumdict.items()):
#    print(ii)
#    print(mtimes)

#
#N_patients = 10
#mu_Y = []
#for ii in range(N_patients): 
#    times = [2,3,4]
#    nonzero_Y = times
#    mu_Y.append(nonzero_Y)
#print(mu_Y)
#mu_Y = np.array(mu_Y).flatten()
#print(mu_Y)
#
#a = [1,2,3]
#print(a[0:-1])
#
#M_ii = np.random.randint(min(3,len(measurement_times)), len(measurement_times)+1)

import random
for ii in range(10):
    measurement_times = [0,10,20,30,40]
    N_remove = np.random.randint(0, len(measurement_times)-2)
    selection_mask = random.sample(measurement_times, N_remove)
    print(selection_mask)
    mtimes_ii = measurement_times[selection_mask]
    print(mtimes_ii)

#        for ii in range(N_patients):
#            times = patient_dictionary[ii].measurement_times
#            nonzero_Y = psi[ii] * (pi_r[ii]*np.exp(rho_r[ii]*times) + (1-pi_r[ii])*np.exp(rho_s[ii]*times))
#            mu_Y.append(nonzero_Y)
#        mu_Y = np.array(mu_Y).flatten()


