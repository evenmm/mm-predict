# Investigate feature importance
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from utilities import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
pd.options.display.width = 150
pd.options.display.max_colwidth = 150

picklefile = open('./binaries_and_pickles/random_forest_model', 'rb')
random_forest_model = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/df_X_covariates', 'rb')
df_X_covariates = pickle.load(picklefile)
picklefile.close()

#Leading X and y from binary_outcome inference
picklefile = open('./binaries_and_pickles/X_test_array_binary_outcome', 'rb')
X_test_array = pickle.load(picklefile)
picklefile.close()
X_test = X_test_array[-1]

picklefile = open('./binaries_and_pickles/y_test_array_binary_outcome', 'rb')
y_test_array = pickle.load(picklefile)
picklefile.close()
y_test = y_test_array[-1]

print("Investigating feature importance based on mean decrease in impurity")
#feature_names = [f"feature {i}" for i in range(df_X_covariates.shape[1])]
feature_names = [col for col in df_X_covariates.columns]

# Feature importance based on mean decrease in impurity (always positive)
start_time = time.time()
importances = random_forest_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest_model.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_names)
forest_importances = pd.DataFrame(forest_importances)
forest_importances = forest_importances.rename({0: 'importances'}, axis=1)  # new method
#print(forest_importances.head(n=5))
forest_importances = forest_importances.sort_values(by=['importances'], ascending=False)
print(forest_importances.head(n=30))

n_features_to_plot = 40
fig, ax = plt.subplots()
forest_importances[0:n_features_to_plot].plot.bar(yerr=std[0:n_features_to_plot], ax=ax)
ax.set_title("Feature importances using mean decrease in impurity")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig("./plots/feature_importance.png")
plt.show()

"""
#Feature importance based on feature permutation
from sklearn.inspection import permutation_importance
print("Running feature importance based on feature permutation...")

start_time = time.time()
result = permutation_importance(
    random_forest_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=15
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
"""
