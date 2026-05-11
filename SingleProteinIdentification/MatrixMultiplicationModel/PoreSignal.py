import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

class PoreSignal: 
    '''Input: local features, global features, weights
       Output: Signal of the nanopore
       Author: Eric Jeanbourquin
       Date: May 11, 2026
       Create a class which we might later use for training the weights and look if a linear model has merit
    '''
    def __init__(self, local_features_file,global_features_file,default_weigths_file,sequence):
        self.weights,self.features= self._load_weights_features_(default_weigths_file)
        self.local_filename = local_features_file
        self.global_filename = global_features_file
        self.sequence = sequence

    def _load_weights_features_(self,path):
        with open(path, 'r') as f:
            file = json.load(f)
            features = {'local':list(file['local'].keys()),'global':list(file['global'].keys())}
            weights = {'local':list(file['local'].values()), 'global': list(file['global'].values())}


        return weights,features
    def create_feature_matrix(self):
        '''To work we need that the local keys and/or global keys to match up with the corresponding
        features in the features file!'''

        # we first create the local features matrix
        with open(self.local_filename,'r') as f:
            local_config = json.load(f)

        local_data = [local_config['residues'][aa] for aa in self.sequence]

        local_feature_matrix = np.array(local_data)   
        # here we create the global matrix if it exist
        if('global' in self.features):
            with open(self.global_filename,'r') as f:
                global_config = json.load(f)

                global_data = [np.array(global_config[feat]) for feat in self.features['global']]
            global_feature_matrix = np.column_stack(global_data)

            full_matrix = np.hstack([local_feature_matrix,global_feature_matrix])
        else: 
            full_matrix = local_feature_matrix
        return full_matrix
    
    def create_toeplitz_matrix(self,matrix_size,window_size:int = 6):
        if window_size > 1:
            hk = window_size // 2
            kernel = np.ones(window_size)
            first_col = np.zeros(matrix_size); first_row = np.zeros(matrix_size)
            first_col[:window_size - hk] = kernel[hk:]
            first_row[:hk + 1] = kernel[:hk + 1][::-1]
            T = toeplitz(first_col, first_row) * (1.0 / window_size)
        return T
    
    def predict(self):
        X = self.create_feature_matrix()
        matrix_size = X.shape[0]
        window_size = 6
        T = self.create_toeplitz_matrix(matrix_size,window_size)
        weights = np.array(self.weights['local']+self.weights['global'])
        
        return T @ X @ weights

            