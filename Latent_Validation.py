#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:09:50 2019

"""

import numpy as np
from sklearn.preprocessing import StandardScaler

def Validation_Score_1D(Latent_Model,Validation_Data,Verbose_Rate=None):
    #Need to standardize validation data before using this function
    
    Observations_Num = len(Validation_Data)
    Features_Num = len(Validation_Data[0])
    
    RSS = 0
    TSS = 0
    for Feature_Index in range(Features_Num):
        
        #Get boolean masks for each group
        Retain_Mask = np.ones((Observations_Num,Features_Num))
        Retain_Mask[:,Feature_Index] = 0
        Holdout_Mask = 1 - Retain_Mask
        
        #Get each data group
        Retain_Data = Validation_Data * Retain_Mask
        Holdout_Data = Validation_Data * Holdout_Mask
        
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
            if (Feature_Index % Verbose_Rate) == 0:
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


def Validation_Score_2D(Latent_Model,Validation_Data):
    #Need to standardize validation data before using this function
    
    Shape = Validation_Data.shape
    if len(Shape) > 3:
        Validation_Data = Validation_Data.reshape(-1,Shape[1],Shape[2])
    
    Observations_Num = len(Validation_Data)
    Features_X_Num = len(Validation_Data[0])
    Features_Y_Num = len(Validation_Data[0,0])
    
    RSS = 0
    TSS = 0
    for Feature_X_Index in range(Features_X_Num):
        for Feature_Y_Index in range(Features_Y_Num):
                    
            #Get boolean masks for each group
            Retain_Mask = np.ones((Observations_Num,Features_X_Num,Features_Y_Num))
            Retain_Mask[:,Feature_X_Index,Feature_Y_Index] = 0
            Holdout_Mask = 1 - Retain_Mask
            
            #Get each data group
            Retain_Data = Validation_Data * Retain_Mask
            Holdout_Data = Validation_Data * Holdout_Mask
            
            for iter in range(2):
                Latent_Scores = Latent_Model.transform(Retain_Data)
                Prediction_Values = Latent_Model.inverse_transform(Latent_Scores)
                Retain_Data[:,Feature_X_Index,Feature_Y_Index] = Prediction_Values[:,Feature_X_Index,Feature_Y_Index]
                continue
            
            #Evaluate performance
            Prediction_Values = Prediction_Values * Holdout_Mask
            RSS += np.sum((Holdout_Data - Prediction_Values)**2)
            TSS += np.sum(Holdout_Data**2)
            
            Q2 = 1 - RSS / TSS
            print(Q2)
                
            continue
        continue
    
    Q2 = 1 - RSS / TSS
    return Q2
