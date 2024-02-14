import inspect
import gc
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from datetime import datetime
import sklearn

import pandas as pd 
from scipy.stats import norm
import pandas as pd
import numpy as np
from numpy import *

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
def preprocess_input(x_raw):
    x=x_raw.copy()
    # Extraction du dixième caractère de la colonne 'SERIE' et création d'une nouvelle colonne 'dixieme_caractere'
    x['dixieme_caractere'] = x['SERIE'].str[9]
    # Création d'un dictionnaire de correspondance des années
    correspondance_annee = {"A": 1980, "B": 1981, "C": 1982, "D": 1983, "E": 1984,
                            "F": 1985, "G": 1986, "H": 1987, "J": 1988, "K": 1989,
                            "L": 1990, "M": 1991, "N": 1992, "P": 1993, "R": 1994,
                            "S": 1995, "T": 1996, "V": 1997, "W": 1998, "X": 1999,
                            "Y": 2000, "1": 2001, "2": 2002, "3": 2003, "4": 2004,
                            "5": 2005, "6": 2006, "7": 2007, "8": 2008, "9": 2009,
                            "A": 2010, "B": 2011, "C": 2012, "D": 2013, "E": 2014,
                            "F": 2015, "G": 2016, "H": 2017, "J": 2018, "K": 2019,
                            "L": 2020, "M": 2021, "N": 2022, "P": 2023, "R": 2024,
                            "S": 2025, "T": 2026, "V": 2027, "W": 2028, "X": 2029,
                            "Y": 2030, "1": 2031, "2": 2032, "3": 2033, "4": 2034,
                            "5": 2035, "6": 2036, "7": 2037, "8": 2038, "9": 2039}

    # Ajout d'une nouvelle colonne "annee_construction" en utilisant la correspondance
    x['annee_construction'] = x['dixieme_caractere'].map(correspondance_annee)

    # Suppression des colonnes indiquées
    cols_to_drop = ["ORDRE", "CNIT", "DESTIN1", "DECISION", "DESTIN_DESCRIPTION_1", 
                    "DESTIN_DESCRIPTION_2", "DESTIN_DESCRIPTION_3", "PROCEDURE_RSV", 
                    "PROCEDURE_MINE", "PROCEDURE_VE", "PROCEDURE_VVR", "PROCEDURE_CG", 
                    "SERIE", "dixieme_caractere", "RAPPORT_EXPERTISE", "DATE_VENTE"]

    x.drop(columns=cols_to_drop, inplace=True)
    
    date_columns_indices = [i for i, col in enumerate(x.columns) if 'DATE' in col]
    for col_index in date_columns_indices:
        x.iloc[:, col_index] = pd.to_datetime(x.iloc[:, col_index])

    # Calcul du nombre d'années écoulées depuis la date actuelle
    
    x['KILOMETRAGE'] = pd.to_numeric(x['KILOMETRAGE'], errors='coerce')
    for col_index in date_columns_indices:
        x.iloc[:, col_index] = (datetime.today() - x.iloc[:, col_index]).dt.days / 365.25
    
    for column in ['CARBURATION', 'MARQUE', 'TYPE_DE_VEHICULE', 'DESTIN2', 'COULEUR', 'MODELE']:
        x[column] = pd.factorize(x[column])[0]
    GPA=x.drop(["DESTIN2"],axis=1)
    X=torch.Tensor(GPA.values).unsqueeze(1).float()
    

    
    # Gpa_sales=Gpa_sales.drop(DESTIN2)
  
    return X

def tensorizer(x):
    GPA=x.drop(["DESTIN2"],axis=1)  
    X=torch.Tensor(GPA.values).unsqueeze(1).float()
    return X

## Définition du modèle
class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model,self).__init__()
        self.fc_block=nn.Sequential(
            nn.Flatten(),
            nn.Linear(17,87),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(negative_slope=0.17),
            
            
            nn.Linear(87,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.17),
            
            
            nn.Linear(128,526),
            nn.BatchNorm1d(526),
            nn.LeakyReLU(negative_slope=0.17),
            
            
            nn.Linear(526,2),
        )
    
    def forward(self,x):
        # x = self.preprocess_input(x)
       
        Lin_out=self.fc_block(x)
        
        return Lin_out.float().unsqueeze(2).unsqueeze(3)

# Importation du modèle entrainé 
model=MLP_model()
model.load_state_dict(torch.load('model.pt'))
model.eval()
# fonction de prédiction de la vente d'un véhicule
def Sales_predictor(Vehicules):
    X=preprocess_input(Vehicules)
    Y=model(X).argmax(axis=1)
    return Y 