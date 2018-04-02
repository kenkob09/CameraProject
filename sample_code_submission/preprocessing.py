from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
try:
    from sklearn.feature_selection import VarianceThreshold
except:
    pass  #ceci pour ne pas avoir une erreur d'importation 
from sklearn.pipeline import Pipeline
from sys import argv, path

from data_manager import DataManager

#L'idee c'est d'utiliser VarianceThreshold, ensuite PCA, comme explique dans la proposition du projet.

class Preprocess(BaseEstimator):
    def __init__(self):
        pca_boi = PCA(n_components=2)
        filtering = VarianceThreshold(threshold = 4.0)
        self.transformer = Pipeline([("first", filtering),("second", pca_boi)])

    def fit(self, X, y=None):
        return self.transformer.fit(X,y)
    
    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X,y)

    def transform(self, X, y=None):
        return self.transformer.transform(X,y)

#Partie test :

if __name__=="__main__":
    
    if len(argv)==1: 
        input_dir = "/anass_coding/Mini-Projet/sample_data" 
        output_dir = "/anass_coding/Mini-Projet/results" 
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'cifar10_model'
    D = DataManager(basename, input_dir) # Load data
    print("*** Before my sexy preprocessing***")
    print D
    
    Prepro = Preprocess()

    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    print("*** Look at these handsome results aye ***")
print D
