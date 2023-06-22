# https://www.datanovia.com/en/lessons/subset-data-frame-rows-in-r/
# https://www.datanovia.com/en/lessons/select-data-frame-columns-in-r/

# Last inn excel-fila med data
library(readxl)
MM_all_patients <- read_excel("MM_all_patients.xls")

# velg bare de variablene vi skal se på for å få det ryddigere
library(dplyr)
MM_all_patients = MM_all_patients %>% select('nnid', 
                                             'Diagnosis date', 'Serum mprotein (SPEP)', 
                                             'Treatment start', 'Serum mprotein (SPEP) (g/l):', 
                                             'Drug 1...99', 'Drug 2...100', 'Drug 3...101', 'Drug 4...102', 'Start date...120', 'End date...121',
                                             'Progression date:', 'Serum mprotein:...141',
                                             'DateOfLabValues', 'SerumMprotein',
                                             'Last-data-entered', 'DateOfDeath')
# Endre navn på to kolonner
colnames(MM_all_patients)[5] <- 'Mprotein_start_of_treatment'
colnames(MM_all_patients)[13] <- 'Mprotein_end_of_treatment'

# Finn gjennomsnitt
mean_start = mean(MM_all_patients$'Mprotein_start_of_treatment', na.rm=TRUE)
mean_end = mean(MM_all_patients$'Mprotein_end_of_treatment', na.rm=TRUE)

difference_after_before = mean_end - mean_start 

# Sjekke ut spredningen i verdier med Box plot
boxplot(MM_all_patients$'Mprotein_start_of_treatment', range=0)
boxplot(MM_all_patients$'Mprotein_end_of_treatment', range=0)

# Beregn variansen 
variance_start = var(MM_all_patients$'Mprotein_start_of_treatment', na.rm=TRUE)
variance_end = var(MM_all_patients$'Mprotein_end_of_treatment', na.rm=TRUE)

# Data for bare pasient nr 1
patient_1_data = MM_all_patients %>% filter(nnid == 1)

