#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:08:04 2019

@author: Jake
"""

import numpy as np
from sklearn.cluster import KMeans


class K_Means():
    
    def __init__(self,k_Centroids=2):
        self.SK_Model = KMeans(n_clusters=k_Centroids)
        self.k_Centroids = k_Centroids
        return
    
    
    def fit(self, X_Train):
        self.SK_Model.fit(X_Train)
        self.X_Train = X_Train
        self.Centroids = self.SK_Model.cluster_centers_
        return


    def transform(self, X, Cluster_Distances=False):
        #Assigns centroid index to each observation in X
        
        Obs_Centroids_Distance_Matrix = np.zeros((len(X),0))
        for Centroid_Index in range(self.k_Centroids):
            
            #Build Observations/Centroids distance matrix
            Centroid_Values = self.Centroids[Centroid_Index]
            Centroid_Obs_Distances = np.sum((X-Centroid_Values)**2,axis=1)
            Centroid_Obs_Distances = Centroid_Obs_Distances.reshape(-1,1)
            Obs_Centroids_Distance_Matrix = np.concatenate((Obs_Centroids_Distance_Matrix,Centroid_Obs_Distances),axis=1)
            continue
        
        #Find minimum distance to centroid for each observation
        Obs_Min_Distances = np.min(Obs_Centroids_Distance_Matrix,axis=1)
        Obs_Min_Distances = Obs_Min_Distances.reshape(-1,1)
        Matrix_Min_Mask = (Obs_Centroids_Distance_Matrix == Obs_Min_Distances)
        Centroid_Assignment_Indices = np.where(Matrix_Min_Mask==True)[1]
        
        #In case only the explicit assignment indices are needed
        if Cluster_Distances == False:
            return Centroid_Assignment_Indices
        
        #Otherwise, return both assignment indices paired with cluster distances      
        return (Centroid_Assignment_Indices, Obs_Min_Distances)

    
    
    def inverse_transform(self,Centroid_Indices):
        #Assigns centroid variable values for each assignment
        
        X_Pred = np.zeros((len(Centroid_Indices),len(self.X_Train[0])))
        for Centroid_Index in range(self.k_Centroids):
            Centroid_Mask = np.where(Centroid_Indices==Centroid_Index)[0]
            X_Pred[Centroid_Mask] = self.Centroids[Centroid_Index]
            continue
            
        return X_Pred