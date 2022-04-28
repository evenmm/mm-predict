import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from matplotlib.patches import Rectangle
import datetime 
from pandas import DataFrame
import seaborn as sns
from color_dictionaries import *
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')
#monthsFmt = mdates.DateFormatter('%b')
#yearsFmt = mdates.DateFormatter('\n\n%Y')  # add some space for the year label
def isNaN(string):
    return string != string
randomnumberr = 4219

# M protein data
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_VISIT_V2.tsv'
print("Loading data frame from file:", filename)
df = pd.read_csv(filename, sep='\t')
print("Number of rows in dataframe:", len(df))
print("PUBLIC_ID of first patient:", df.loc[0,['PUBLIC_ID']][0])
print("PUBLIC_ID of last patient:", df.loc[len(df)-1,['PUBLIC_ID']][0])
print(df.head(n=5))
#for col in df.columns:
#    print(col)
# Columns with dates, drugs or mproteins
df_mprotein_and_dates = df[['PUBLIC_ID', 'VISITDY',
'D_LAB_serum_m_protein',
#'D_IM_LIGHT_CHAIN_BY_FLOW', #"kappa" or "lambda"
#'D_LAB_serum_kappa', # Serum Kappa (mg/dL)
#'D_LAB_serum_lambda', # Serum Lambda (mg/dL)
#'D_IM_kaplam'
]]

# Remove lines with nan times 
df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['VISITDY'].notna()]
df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['D_LAB_serum_m_protein'].notna()]
df_mprotein_and_dates.reset_index(drop=True, inplace=True)

print(df_mprotein_and_dates.head(n=5))

# Define outcome y based on last observation:
# Outcome is defined as last M protein value 
# Remove last observation from covariate dataset

# create new dataframe 
PUBLIC_ID_list = pd.unique(df_mprotein_and_dates[['PUBLIC_ID']].values.ravel('K'))

y2 = pd.DataFrame(
    {'PUBLIC_ID' : PUBLIC_ID_list}
)
y2["outcome"] = np.nan
print(y2.head(n=5))
y = []
y_classify = [] # 1: M protein stays the same or goes down, 0: M protein goes up 
Y_rolled = []
y_classify_rolled = []
time_to_prediction = []

# Iterate the dataframe and give the y matrix the value of the last M protein by assigning the value in each row
#prev_PUBLIC_ID = df_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
#prev_M_protein_value = df_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
PUBLIC_ID = np.nan 
prev_M_protein_value = np.nan 
prev_M_protein_time = np.nan 
Mproteincount = 0
y2index = 0
df_mprotein_and_dates["y2index"] = np.nan
for index, row in df_mprotein_and_dates.iterrows():
    if not row['PUBLIC_ID'] == PUBLIC_ID:
        # New public ID, if there is more than 1 measurement of the last one, assign the last m protein value to PUBLIC_ID in y. 
        if Mproteincount > 1:
            y.append(prev_M_protein_value)
            time_to_prediction.append(row['VISITDY'] - prev_M_protein_time)
            y_classify.append(int(row['D_LAB_serum_m_protein'] <= prev_M_protein_value))
            y2.loc[y2['PUBLIC_ID'] == PUBLIC_ID, ['outcome']] = prev_M_protein_value
            y2index = y2index + 1
            # Mark the last value for that patient for deletion 
            # must be deep copy!! 
            df_mprotein_and_dates.loc[index-1, ['D_LAB_serum_m_protein']] = np.nan
            #df.loc[df[<some_column_name>] == <condition>, [<another_column_name>]] = <value_to_add>
        
        PUBLIC_ID = row['PUBLIC_ID']
        Mproteincount = 0
    if Mproteincount > 1: 
        Y_rolled.append(row['D_LAB_serum_m_protein'])
        y_classify_rolled.append(int(row['D_LAB_serum_m_protein'] <= prev_M_protein_value))
    df_mprotein_and_dates.loc[index, ['y2index']] = y2index
    prev_M_protein_value = row['D_LAB_serum_m_protein']
    prev_M_protein_time = row['VISITDY']
    Mproteincount = Mproteincount + 1 
if Mproteincount > 1:
    Y_rolled.append(row['D_LAB_serum_m_protein'])
    y_classify_rolled.append(int(row['D_LAB_serum_m_protein'] <= prev_M_protein_value))
    y.append(prev_M_protein_value)
    time_to_prediction.append(row['VISITDY'] - prev_M_protein_time)
    y_classify.append(int(row['D_LAB_serum_m_protein'] <= prev_M_protein_value))
    y2.loc[y2['PUBLIC_ID'] == PUBLIC_ID, ['outcome']] = prev_M_protein_value
    # Mark the last value for that patient for deletion 
    # must be deep copy!! 
    df_mprotein_and_dates.loc[index, ['D_LAB_serum_m_protein']] = np.nan
    #df.loc[df[<some_column_name>] == <condition>, [<another_column_name>]] = <value_to_add>
# Erase the outcome values in the history df
print(df_mprotein_and_dates.head(n=30))
### Print entire df
##with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
##    print(df_mprotein_and_dates)
df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['D_LAB_serum_m_protein'].notna()]
df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['y2index'].notna()]
df_mprotein_and_dates.reset_index(drop=True, inplace=True)

#print(y2.head(n=5))
y2 = y2['outcome'].squeeze()
print("print(y2.head(n=5))")
print(y2.head(n=5))
y = np.array(y)
y_classify = np.array(y_classify)
print("Average y 0/1 value:", np.mean(y_classify))
print(y[0:5])
print("len(y):", len(y))
print("len(y2):", len(y2))
print("len df:", len(pd.unique(df_mprotein_and_dates[['PUBLIC_ID']].values.ravel('K'))))

# "Roll" dataframe to use all the available nested data as training points
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
X_rolled, Y_rolled = make_forecasting_frame(df_mprotein_and_dates["D_LAB_serum_m_protein"], kind="Mprotein", max_timeshift=1, rolling_direction=1)
print(X_rolled.head(n=30))
print(Y_rolled.head(n=30))
#X_rolled = roll_time_series(df_mprotein_and_dates, column_id="y2index", column_sort="VISITDY")
#print(X_rolled.head(n=30))
print("len(X_rolled)", len(X_rolled))
print("len(Y_rolled)", len(Y_rolled))
print("len(y_classify_rolled)", len(y_classify_rolled))

# Feature extraction
from tsfresh import extract_features
#X_rolled = X_rolled[["id", "D_LAB_serum_m_protein", "VISITDY"]]

extracted_features = extract_features(X_rolled, column_id="id",
                    column_sort="time", column_value="value",
                    n_jobs=0, default_fc_parameters={"maximum": None},
                    disable_progressbar=True)
#extracted_features = extract_features(X_rolled, column_id="id", column_sort="VISITDY") #y2index can be removed 
#print("len(extracted_features)", len(extracted_features))
print(extracted_features.head(n=100))

# Impute nan and inf
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features)

# Here we add other predictors to the df!
#extracted_features["time_to_prediction"] = time_to_prediction
# Add the drugs: 






# Split into train and test 
from sklearn.model_selection import train_test_split
X_full_train, X_full_test, y_train, y_test = train_test_split(X_rolled, Y_rolled, test_size=.4, random_state=randomnumberr) #random_state=randomnumberr

# Train a random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
random_forest_model = RandomForestRegressor(random_state=randomnumberr)
random_forest_model.fit(X_full_train, y_train)
#print(classification_report(y_test, random_forest_model.predict(X_full_test)))
print("R2 score, regression:", r2_score(y_test, random_forest_model.predict(X_full_test)))

# Classification random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_full_train, X_full_test, y_train, y_test = train_test_split(X, y_classify_rolled, test_size=.4, random_state=randomnumberr)
classifier_full = RandomForestClassifier(random_state=randomnumberr)
classifier_full.fit(X_full_train, y_train)
print("Classification:", classification_report(y_test, classifier_full.predict(X_full_test)))

from sklearn.metrics import plot_roc_curve
plot_roc_curve(classifier_full, X_full_test, y_test) 
plt.show()
