'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
from __future__ import print_function

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import sklearn

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import  KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

#from visualisation import Visualisation # groupe visu module
from preprocessing import Preprocess
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        #self.mod = MultinomialNB()
        self.mod = Pipeline([('what', Preprocess()),('a pain',MultinomialNB(alpha=0.05, fit_prior=False, class_prior = None))])

    def define_model(self, name):
        if self.is_trained == False:
            if name == 'preprocInc':
                #self.mod = MultinomialNB()
                self.mod = Pipeline([('what', Preprocess()),('a pain',MultinomialNB(alpha=0.05, fit_prior=False, class_prior = None))])
            else:
                print('Error selecting the model, choose by default Gaussian NB')
                self.mod = GaussianNB()
        else:
            print("Model already load")

    def fit(self, X, Y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''


        # For multi-class problems, convert target to be scikit-learn compatible
        # into one column of a categorical variable
        y=self.convert_to_num(Y, verbose=False)

        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1] # Does not work for sparse matrices
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        self.mod.fit(X, y)
        self.is_trained=True
        
        result = self.mod.predict(X)
        print("Precision du modele sur l'ensemble d'apprentissage:",accuracy_score(result, y)*100,"%") 
        
        print(__doc__)
        
        # VISUALISATION
        #visu = Visualisation()
        #visu.plot(self.mod, (X, y), self.convert_to_num(self.predict(X), verbose=False))
        
        #Confusion Matrix
        print(confusion_matrix(y, result))
        
        
    def fit_score_and_find_best_param(self, X, Y):
        
        y=self.convert_to_num(Y, verbose=False) 
        
        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=0)

        
        self.fit(X_train,y_train)
        result = self.mod.predict(X_test)
        print("Precision du modele sur l'ensemble de test:",accuracy_score(result, y_test)*100,"%") 
        
        
        #Exhaustive Grid Search
        
        tuned_parameters = [{'fit_prior': [True], 'class_prior': [None],'alpha': [1, 0.1, 0.05,0.037, 0.025, 0.01, 0.001, 0.0001]},
                        {'fit_prior': [False], 'class_prior': [None], 'alpha': [1, 0.1, 0.05,0.037, 0.025, 0.01, 0.001, 0.0001]},
                        {'fit_prior': [True], 'class_prior': [[0.10 for i in range(10)]],'alpha': [1, 0.1, 0.05,0.037, 0.025, 0.01, 0.001, 0.0001]},
                        {'fit_prior': [False], 'class_prior': [[0.10 for i in range(10)]], 'alpha': [1, 0.1, 0.05,0.037, 0.025, 0.01, 0.001, 0.0001]}]
        
        scores = ['precision', 'recall']
    
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
        
            clf = GridSearchCV(self.mod, tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
        
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()    


    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))

        # Return predictions as class probabilities
        y = self.mod.predict_proba(X)
        return y

    def save(self, path="./"):
        with open(path + '_model.pickle', 'wb') as f:
            print('modele name : ', path + '_model.pickle')
            pickle.dump(self , f)

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

    def convert_to_num(self, Ybin, verbose=True):
        ''' Convert binary targets to numeric vector (typically classification target values)'''
        if verbose: print("Converting to numeric vector")
        Ybin = np.array(Ybin)
        if len(Ybin.shape) ==1: return Ybin
        classid=range(Ybin.shape[1])
        Ycont = np.dot(Ybin, classid)
        if verbose: print(Ycont)
        return Ycont


#Tests Unitaires

from sklearn.base import  BaseEstimator

def convert_to_num( Ybin, verbose=True):
        ''' Convert binary targets to numeric vector (typically classification target values)'''
        if verbose: print("Converting to numeric vector")
        Ybin = np.array(Ybin)
        if len(Ybin.shape) ==1: return Ybin
        classid=range(Ybin.shape[1])
        Ycont = np.dot(Ybin, classid)
        if verbose: print(Ycont)
        return Ycont   

class Predictor(BaseEstimator):
    '''Predictor: modify this class to create a predictor of
    your choice. This could be your own algorithm, of one for the scikit-learn
    models, for which you choose the hyper-parameters.'''
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mod = MultinomialNB()
        print("PREDICTOR=" + self.mod.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mod = self.mod.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mod.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
class Predictor_with_params(BaseEstimator):
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mod = MultinomialNB(alpha=0.05, fit_prior=False, class_prior = None)
        print("PREDICTOR=" + self.mod.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mod = self.mod.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mod.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        
from sys import argv, path       

if __name__=="__main__":
    # Modify this class to serve as test
    
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data" # A remplacer par le bon chemin
        output_dir = "../results" # A remplacer par le bon chemin
        code_dir = "../ingestion_program" # A remplacer par le bon chemin
        metric_dir = "../scoring_program" # A remplacer par le bon chemin
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        code_dir = argv[3]
        metric_dir = argv[4]
        
    path.append (code_dir)
    path.append (metric_dir)
    
    metric_name = 'bac_multiclass'
    import libscores
    scoring_function = getattr(libscores, metric_name)
    print('Using scoring metric:', metric_name)
            
    from data_manager import DataManager    
    basename = 'cifar10'
    D = DataManager(basename, input_dir) # Load data
    print(D)
    
    # Here we define two models and compare them; you can define more than that
    model_dict = {
            'Multinomial sans preprocess': Predictor(),
            'Multinomial sans preprocess et avec hyper parametres optimaux': Predictor_with_params(),
            'Multinomial avec Preprocess': Pipeline([('prepro', Preprocess()), ('Mutlinomial', Predictor())]),
            'Multinomial avec Preprocess et avec hyper parametres optimaux': Pipeline([('prepro', Preprocess()), ('Mutlinomial', Predictor_with_params())])
            }
        
    for key in model_dict:
        mymodel = model_dict[key]
        print("\n\n *** Model {:s}:{:s}".format(key,model_dict[key].__str__()))
 
        # Train
        print("Training")
        X_train = D.data['X_train']
        Y_train = D.data['Y_train']
        y_train= convert_to_num(Y_train, verbose=False)
        mymodel.fit(X_train, y_train)
    
        # Predictions on training data
        print("Predicting")
        Ypred_tr = mymodel.predict(X_train)
        
        # Cross-validation predictions
        print("Cross-validating")
        from sklearn.model_selection import KFold
        from numpy import zeros  
        n = 10 # 10-fold cross-validation
        kf = KFold(n_splits=n)
        kf.get_n_splits(X_train)
        Ypred_cv = zeros(Ypred_tr.shape)
        i=1
        for train_index, test_index in kf.split(X_train):
            print("Fold{:d}".format(i))
            Xtr, Xva = X_train[train_index], X_train[test_index]
            Ytr, Yva = Y_train[train_index], Y_train[test_index]
            ytr= convert_to_num(Ytr, verbose=False)
            mymodel.fit(Xtr, ytr)
            Ypred_cv[test_index] = mymodel.predict(Xva)
            i = i+1
            

        # Compute and print performance
        
        #training_score = scoring_function(Y_train, Ypred_tr)
        #cv_score = scoring_function(Y_train, Ypred_cv)
        
        training_score = accuracy_score(Ypred_tr, y_train)*100
        cv_score = accuracy_score (Ypred_cv, y_train)*100
        
        
        print("\nRESULTS FOR SCORE {:s}".format(metric_name))
        print("TRAINING SCORE= {:f}".format(training_score))
        print(confusion_matrix(y_train, Ypred_tr))
        print("CV SCORE= {:f}".format(cv_score))