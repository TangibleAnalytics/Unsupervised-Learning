#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:09:50 2019

@author: Jake
"""

import numpy as np


class Latent_Validator():
    
    def __init__(self):
        return
    
    
    def Validation_Score(self, Latent_Model, X_Test, Sample_Fraction=1, Verbose_Rate=None):
        #Need to standardize validation data before using this function
        #Latent Model must have both transform() and inverse_transform() methods
        
        Observations_Num = len(X_Test)
        Features_Num = len(X_Test[0])
        
        #Sample a subset of full feature space for purpose of time savings
        Sample_Indices = self.__Allocated_Indices(Sample_Fraction=Sample_Fraction, Full_Length=Features_Num)
        
        RSS = 0
        TSS = 0
        for Feature_Index in Sample_Indices:
            
            #Get boolean masks for each group
            Retain_Mask = np.ones((Observations_Num,Features_Num))
            Retain_Mask[:,Feature_Index] = 0
            Holdout_Mask = 1 - Retain_Mask
            
            #Get each data group
            Retain_Data = X_Test * Retain_Mask
            Holdout_Data = X_Test * Holdout_Mask
            
            #Get prediction values
            for iter in range(2):
                Latent_Scores = Latent_Model.transform(Retain_Data)
                Prediction_Values = Latent_Model.inverse_transform(Latent_Scores)
                Retain_Data[:,Feature_Index] = Prediction_Values[:,Feature_Index]
                continue
            
            #Evaluate performance
            Prediction_Values = Prediction_Values * Holdout_Mask
            RSS += np.sum((Holdout_Data - Prediction_Values)**2)
            TSS += np.sum(Holdout_Data**2)
            
            #Progress update
            if Verbose_Rate == None:
                continue
            else:
                Verbose_Score = np.random.uniform(low=0,high=1)
                if Verbose_Score < Verbose_Rate:
                    Q2 = 1 - RSS / TSS
                    Q2 = round(Q2,3)
                    print('CV Q2 Score: '+str(Q2))
                    Progress_Proportion = (Feature_Index+1) / Features_Num
                    Progress_Proportion = round(100*Progress_Proportion,1)
                    print('Percent Complete: '+str(Progress_Proportion))
                    print('\n')
            
            continue
        
        Q2 = 1 - RSS / TSS
        return Q2
    
    
    def __Allocated_Indices(self, Sample_Fraction, Full_Length):
        #Grab a predetermined subset of feature indices for testing purposes
        
        Random_Indices = []
        while len(Random_Indices) < (Sample_Fraction * Full_Length):
        
            Random_Index = np.random.randint(low=0, high=Full_Length)
            while Random_Index in Random_Indices:
                Random_Index = np.random.randint(low=0, high=Full_Length)
                continue
            
            Random_Indices.append(Random_Index)
            continue
        Random_Indices = np.array(Random_Indices)
        Random_Indices.sort()
        return Random_Indices

