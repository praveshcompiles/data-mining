import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#df_train = pd.read_csv("training.csv",header=0,index_col=0)
df = pd.read_csv("dataset.csv",header=0,index_col=0)

'''Normalize data'''
normvalue = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
normvalue_scaled = min_max_scaler.fit_transform(normvalue)
df_new = pd.DataFrame(normvalue_scaled,columns=df.columns, index=df.index).reset_index()
print(df_new)
classnames = df_new.groupby(["class"]).count().reset_index()['class'].values.tolist()

# '''print the top features names'''
# Y1 = df_new['class']
# X1 = pd.DataFrame(df_new.drop(['class'], axis=1).values, columns=df_new.drop(['class'], axis=1).columns)
# select_model= SelectKBest(chi2, k=60).fit(X1,Y1)
# selected_feature_names=X1.columns[select_model.get_support(indices=True)]
# print('Feature list:', selected_feature_names)

'''Transform class name to numeric'''
df_new[["class"]]=df_new[["class"]].apply(LabelEncoder().fit_transform)

array = df_new.values
X = array[:,1:148]
Y = array[:,0]


'''find and print best scored features'''
# chi2 calculates the correlation between features and labels, and select the most related feature
X_select_feature = SelectKBest(chi2, k=60).fit_transform(X, Y)
X_train = X_select_feature[0:168,:]
Y_train = array[0:168,0]
X_test = X_select_feature[168:,:]
Y_test = array[168:,0]



'''Improve classifier, set hyperparameters for RandomForest'''
# try out a wide range of values
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier(random_state = 42)
# Random search of parameters, using 5 fold cross validation to split the training data
# search across 100 different combinations, and use all available cores
# The most important arguments in RandomizedSearchCV are n_iter,
# which controls the number of different combinations to try
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error',
                              cv = 5, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=False)
rf_random.fit(X_train, Y_train)
print("The best parameter:", rf_random.best_params_)


def evaluate(model, test_features, test_labels):
    predictions1 = model.predict(test_features)
    errors = abs(predictions1 - test_labels)

    accuracy = accuracy_score(test_labels,predictions1)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy =',accuracy)
    return accuracy



'''classification and  evaluation metrics'''
clf = RandomForestClassifier()
#clf = GaussianNB()
trained_model = clf.fit(X_train, Y_train)
predictions = trained_model.predict(X_test)
base_accuracy = accuracy_score(Y_test, predictions)
print("Trained model : ", trained_model)
print("Train Accuracy : ", accuracy_score(Y_train, trained_model.predict(X_train)))
print("Test Accuracy  : ", base_accuracy)
print(classification_report(Y_test,predictions,digits=3))
print("Test confusion matrix :\n", confusion_matrix(Y_test, predictions))
print("Trained confusion matrix :\n", confusion_matrix(Y_train, trained_model.predict(X_train)))

#base_model = RandomForestClassifier()
#base_model.fit(X_train, Y_train)
#base_accuracy = evaluate(base_model, X_test, Y_test)
#print(rf_random.best_estimator_, rf)
best_model = rf_random.best_estimator_
best_accuracy = evaluate(best_model, X_test, Y_test)
print(classification_report(Y_test,best_model.predict(X_test)))
print('Improvement of {:0.2f}%.'.format( 100 * (best_accuracy - base_accuracy) / base_accuracy))

'''Visualize the confusion matrix'''
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #fmt = '%d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cnf_matrix1 = confusion_matrix(Y_test, predictions)
#cnf_matrix = confusion_matrix(Y_train, trained_model.predict(X_train))
plt.figure()
plot_confusion_matrix(cnf_matrix1, classes=classnames,
                      title='confusion matrix')

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, best_model.predict(X_test))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classnames,
                      title='confusion matrix after hyperparameter tuning')


plt.show()