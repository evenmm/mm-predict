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
