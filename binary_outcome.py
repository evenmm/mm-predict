# Take history regions and estimates
# Extract features and learn the mapping from features to drug response parameters
from utilities import *
from sklearn import preprocessing
from sklearn import utils
start_time = time.time()
warnings.simplefilter("ignore")

# Load binary outcome Y
picklefile = open('./binaries_and_pickles/Y_increase_or_not', 'rb')
Y_increase_or_not = pickle.load(picklefile)
lab = preprocessing.LabelEncoder()
Y_increase_or_not = lab.fit_transform(Y_increase_or_not)
picklefile.close()
print("Number of intervals:", len(Y_increase_or_not))
print("Number of 0s:", len(Y_increase_or_not) - np.count_nonzero(Y_increase_or_not))
print("Number of 1s:", sum(Y_increase_or_not[Y_increase_or_not == 1]))
#print("Number of nans:", sum(np.isnan(Y_increase_or_not)))
#print("Number of other things:", sum([(elem not in [0,1]) for elem in Y_increase_or_not]) - sum(np.isnan(Y_increase_or_not)))

# Load covariate dataframe X
picklefile = open('./binaries_and_pickles/df_X_covariates', 'rb')
df_X_covariates = pickle.load(picklefile)
picklefile.close()

######################################################################
# Split into train and test 
######################################################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
# Train a random forest regressor
randomnumberr = 4219

r2score_train_array = []
r2score_test_array = []
print("Training and testing")

def train_random_forest(args):
    df_X_covariates, Y_outcome, i = args

    random_forest_model = RandomForestClassifier(n_estimators=1000, random_state=randomnumberr)
    X_train, X_test, y_train, y_test = train_test_split(df_X_covariates, Y_outcome, test_size=.4, stratify=Y_outcome, random_state=randomnumberr + i)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    y_pred_train = random_forest_model.predict(X_train)

    # calculate the fpr and tpr for all thresholds of the classification
    probs = random_forest_model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # train data too
    probs_train = random_forest_model.predict_proba(X_train)
    preds_train = probs_train[:,1]
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, preds_train)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)

    return (random_forest_model, X_train, X_test, y_train, y_test, y_pred, y_pred_train, probs, preds, fpr, tpr, roc_auc, probs_train, preds_train, fpr_train, tpr_train, threshold_train, roc_auc_train)

all_random_states = range(50)
args = [(df_X_covariates, Y_increase_or_not, i) for i in all_random_states]
with Pool(15) as pool:
    random_forest_model_array, X_train_array, X_test_array, y_train_array, y_test_array, y_pred_array, y_pred_train_array, probs_array, preds_array, fpr_array, tpr_array, roc_auc_array, probs_train_array, preds_train_array, fpr_train_array, tpr_train_array, threshold_train_array, roc_auc_train_array = zip(*pool.map(train_random_forest, args))
    #results = pool.map(train_random_forest, args)

#model_array = [elem[0] for elem in results]
#X_train_array = [elem[1] for elem in results]
#X_test_array = [elem[2] for elem in results]
#y_train_array = [elem[3] for elem in results]
#y_test_array = [elem[4] for elem in results]
#y_pred_array = [elem[5] for elem in results]
#y_pred_train_array = [elem[6] for elem in results]
#roc_auc_array = 

#model = model_array[-1]
#X_train = X_train_array[-1]
#X_test = X_test_array[-1]
#y_test = y_test_array[-1]
#y_train = y_train_array[-1]
#y_pred = y_pred_array[-1]
#y_pred_train = y_pred_train_array[-1]

random_forest_model = random_forest_model_array[-1]
X_train = X_train_array[-1]
X_test = X_test_array[-1]
y_train = y_train_array[-1]
y_test = y_test_array[-1]
y_pred = y_pred_array[-1]
y_pred_train = y_pred_train_array[-1]
probs = probs_array[-1]
preds = preds_array[-1]
fpr = fpr_array[-1]
tpr = tpr_array[-1]
roc_auc = roc_auc_array[-1]
probs_train = probs_train_array[-1]
preds_train = preds_train_array[-1]
fpr_train = fpr_train_array[-1]
tpr_train = tpr_train_array[-1]
threshold_train = threshold_train_array[-1]
roc_auc_train = roc_auc_train_array[-1]

end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)
print("Average AUC score on train data:", np.mean(roc_auc_train_array), "std:", np.std(roc_auc_train_array))
print("Average AUC score on test data:", np.mean(roc_auc_array), "std:", np.std(roc_auc_array))

# method I: plt
plt.title('Receiver Operating Characteristic, test data')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(visible=True)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("./plots/ROC_test_data.png")
#plt.show()

print(classification_report(y_test, y_pred))

picklefile = open('./binaries_and_pickles/random_forest_model', 'wb')
pickle.dump(random_forest_model, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/X_test_array_binary_outcome', 'wb')
pickle.dump(X_test_array, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/y_test_array_binary_outcome', 'wb')
pickle.dump(y_test_array, picklefile)
picklefile.close()
