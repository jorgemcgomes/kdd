import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Data preparation ##################################

def dataload(name,folder='/mnt/ssd/kdd/'):
    X = pd.read_csv(folder + 'train_x_'+ name + '.csv')
    X.drop('projectid', axis=1, inplace=True)

    X_master = pd.read_csv(folder + 'test_x_'+ name + '.csv')
    X_master.drop('projectid', axis=1, inplace=True)

    # get dummy variables for the categorical features
    # joining the train with the test data to ensure that the
    # one-hot-encoding is consistent in both
    # this should be redone using one OneHotEncoder per categorical feature
    joined = pd.concat((X, X_master), axis=0)
    joined = pd.get_dummies(joined, drop_first=True)

    # Now splitting them again
    X_master = joined.iloc[-X_master.shape[0]:]
    X = joined.iloc[:-X_master.shape[0]]
    y = pd.read_csv(folder + 'train_y_' + name + '.csv')
    y = y['is_exciting']

    # Double-check that are no NaN values anywhere
    if len(X[X.isnull().T.any().T]) > 0:
        raise Exception("NaN values in X")
    if len(X_master[X_master.isnull().T.any().T]) > 0:
        raise Exception("NaN values in X test")

    return X, y, X_master

def splitdata(X, y, test_split=0.2):
    # split into train and test
    lim = int((1 - test_split) * X.shape[0])
    # the test data is the most recent data
    # we're trying to predict future outcomes, so this is the most accurate model assessment
    X_train = X.iloc[:lim, ]
    y_train = y.iloc[:lim, ]
    X_test = X.iloc[lim:, ]
    y_test = y.iloc[lim:, ]
    return X_train, y_train, X_test, y_test

def featurescaling(X, sc=None):
    # Feature Scaling: mean and stdev scaling
    if sc==None:
        sc = StandardScaler()
        X = sc.fit_transform(X)
    else:
        X = sc.transform(X)
    if not np.isfinite(X.all()):
        raise Exception('Found non-finite values')

    return X, sc

def load_and_transform(name, folder='/mnt/ssd/kdd/', split=0.2):
    X, y, Xm = dataload(name, folder=folder)
    Xtr, ytr, Xte, yte = splitdata(X, y, test_split=split)
    # only using the train data for computing the standardization coefficients
    Xtr, sc = featurescaling(Xtr)
    if len(Xte) > 0:
        Xte, _ = featurescaling(Xte, sc)
    Xm, _ = featurescaling(Xm, sc)
    return Xtr, ytr, Xte, yte, Xm

def assessmodel(model, X_test, y_test):
    '''
    We assess the model in the test data (the most recent)
    This is representative of the a production environment, where we
    use the past data to predict future projects
    Cross-validation does not make much sense in this scenario, as
    we would be using new projects to predict old projects
    And it is also computationally expensive
    '''
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = y_prob > 0.5 # cutoff
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n' + str(cm))
    # True Positive Rate: When it's actually yes, how often does it predict yes?
    tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # False Positive Rate: When it's actually no, how often does it predict yes?
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    # Precision: When it predicts yes, how often is it correct?
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    print('True Positive Rate:', round(tpr, 2))
    print('False Positive Rate:', round(fpr, 2))
    print('Precision:', round(precision, 2))
    auc = roc_auc_score(y_test, y_prob)
    print('ROC AUC score: ' + str(auc))
    # ROC curve
    fpr,tpr,_ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def fitLRmodel(X, y):
    from sklearn.linear_model import LogisticRegression
    # Balanced class_weight because we have way more non-exciting than exciting
    lr = LogisticRegression(penalty='l1', class_weight='balanced', verbose=1)
    lr.fit(X, y)
    return lr


X_train, y_train, X_test, y_test, X_master = load_and_transform('basic', split=0.2)
lr1 = fitLRmodel(X_train, y_train)
assessmodel(lr1, X_test, y_test)

X_train, y_train, X_test, y_test, X_master = load_and_transform('feature_eng_simple', split=0.2)
lr2 = fitLRmodel(X_train, y_train)
assessmodel(lr2, X_test, y_test)

X_train, y_train, X_test, y_test, X_master = load_and_transform('feature_eng_census', split=0.2)
lr3 = fitLRmodel(X_train, y_train)
assessmodel(lr3, X_test, y_test)

X_train, y_train, X_test, y_test, X_master = load_and_transform('feature_eng_nlp', split=0.2)
lr4 = fitLRmodel(X_train, y_train)
assessmodel(lr4, X_test, y_test)

# For the Kaggle competition
X_train, y_train, _, _, X_master = load_and_transform('feature_eng_nlp', split=0)
lrfinal = fitLRmodel(X_train, y_train)
pred = lrfinal.predict_proba(X_master)[:,1]

# Write submission file
ori = pd.read_csv('/mnt/ssd/kdd/test_x_feature_eng_nlp.csv') # get the projectids
out_master = pd.DataFrame({'projectid':ori['projectid'].values,'is_exciting':pred})
out_master= out_master[['projectid','is_exciting']] # sort the columns
out_master.to_csv('data/kaggle.csv', sep=',', index=False) # write the submission file


# Deep Neural Networks

from tensorflow.python.keras.callbacks import Callback
class ROCMetric(Callback):
    """
    Custom metric to evaluate the ROC AUC at the end of each epoch in
    the validation data
    """
    def __init__(self, validation_data=(), interval=1):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logs['rocauc'] = score
            print('ROC AUC: ' + str(score))

def fitDNN(X_train, y_train, X_test, y_test):
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Dropout

    # Initialise the ANN
    ann = Sequential()
    # input layer and first hidden layer
    ann.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu', input_dim=X_train.shape[1]))
    ann.add(Dropout(rate=0.4))
    # second hidden layer
    ann.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
    ann.add(Dropout(rate=0.4))
    # output layer, only one output node: probability of being exciting
    ann.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    # compiling the ANN -- applying Stochastic Gradient Descent
    ann.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    print(ann.summary())

    roc = ROCMetric(validation_data=(X_test, y_test), interval=1)
    # fitting the network to data
    # note that this returns an history object with the cost, accuracy over the training
    from sklearn.utils import class_weight
    class_w = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))
    h = ann.fit(X_train, y_train, batch_size=32, epochs=100, class_weight=class_w,
                validation_data=(X_test,y_test), callbacks=[roc], verbose=2)
    return ann, h


ann4, history4 = fitDNN(X_train, y_train, X_test, y_test)
assessmodel(ann4, X_test, y_test)

plt.plot(history4.history['acc'], label='Train Acc')
plt.plot(history4.history['val_acc'], label='Val Acc')
plt.plot(history4.history['rocauc'], label='Val ROC-AUC')
plt.legend()
plt.show()