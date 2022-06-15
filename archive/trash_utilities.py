#def get_full_Mprotein_history(params, measurement_times, treatment_history, buffer=10):
#    # Get values of M protein from time 0 until 10 days after last treatment
#    last_time = measurement_times[-1]+buffer
#    all_times = np.linspace(0, last_time, len(measurement_times)+buffer)
#    
#    Y_t = params.Y_0
#    for treat_index in range(len(treatment_history)):
#        # Find the correct drug effect k_D
#        if treatment_history[treat_index].id == 0:
#            drug_effect = 0
#        elif treatment_history[treat_index].id == 1:
#            drug_effect = params.k_D
#        else:
#            print("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
#            sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
#        # Calculate the M protein value at the end of this treatment line
#        Y_t = Y_t * params.pi_r * np.exp(params.g_r * measurement_times) + Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * measurement_times)        
#    return Y_t
