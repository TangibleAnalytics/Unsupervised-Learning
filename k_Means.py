#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:08:04 2019

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



class K_Means():
    
    
    def __init__(self,k_Centroids=2):
        self.k_Centroids = k_Centroids
        return
    
    
    def fit(self,X_Train):
        
        #Store raw training data
        self.X_Train_Raw = X_Train
        
        #Store scaled training data as well as scaler model
        self.Scaler = StandardScaler()
        self.Scaler.fit(self.X_Train_Raw)
        self.X_Train_Scaled = self.Scaler.transform(self.X_Train_Raw)
        
        #Sort data observations according to distance from global centroid value
        Obs_Distances = np.sum(self.X_Train_Scaled**2,axis=1)
        Obs_Index_Distances = []
        for Obs_Index in range(len(self.X_Train_Scaled)):
            Obs_Index_Distances.append([Obs_Index,Obs_Distances[Obs_Index]])
            continue
        
        #Sort ascending
        Obs_Index_Distances.sort(key = lambda x: x[1], reverse=False)
        Obs_Index_Distances = np.array(Obs_Index_Distances)
        
        #Get top k centroid indices, convert float to int, take top observations
        Initial_Centroid_Indices = Obs_Index_Distances[:self.k_Centroids,0]
        Initial_Centroid_Indices = np.array(Initial_Centroid_Indices,dtype='int')
        self.Centroids = self.X_Train_Scaled[Initial_Centroid_Indices]
        
        
        #Calibrate centroids
        Previous_Assignment_Indices = 0
        while True:
            
            #Get current centroid index assignments
            Centroid_Assignment_Indices = self.transform(X=self.X_Train_Scaled,Scale=False)
            
            #Termination condition
            if (Centroid_Assignment_Indices == Previous_Assignment_Indices).all():
                break
            
            #Update current centroid values
            for Centroid_Index in range(self.k_Centroids):
                
                Centroid_Mask = np.where(Centroid_Assignment_Indices==Centroid_Index)[0]
                Centroid_Obs = self.X_Train_Scaled[Centroid_Mask]
                Centroid_Values = np.mean(Centroid_Obs,axis=0)
                self.Centroids[Centroid_Index] = Centroid_Values
                continue
            
            #Update value of next loop's previous index assignments
            Previous_Assignment_Indices = np.copy(Centroid_Assignment_Indices)            
            continue
        return
    
    
    def transform(self,X,Scale=False):
        #Assigns centroid index to each observation in X
        
        if Scale == True:
            X = self.Scaler.transform(X)
        
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
        
        #In case explicit assignment indices are needed
        Centroid_Assignment_Indices = np.where(Matrix_Min_Mask==True)[1]
        return Centroid_Assignment_Indices
    
    
    def inverse_transform(self,Centroid_Indices,Scale=False):
        #Assigns centroid variable values for each assignment
        
        X_Pred = np.zeros((len(Centroid_Indices),len(self.X_Train_Scaled[0])))
        for Centroid_Index in range(self.k_Centroids):
            Centroid_Mask = np.where(Centroid_Indices==Centroid_Index)[0]
            X_Pred[Centroid_Mask] = self.Centroids[Centroid_Index]
            continue
            
        #Rescale data
        if Scale == True:
            X_Pred = self.Scaler.inverse_transform(X_Pred)
            
        return X_Pred
    




