# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
import numpy as np
import torch
import os
from ecgdetectors import Detectors

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  

    FILE = 'model.pth'

    # prediction model
    def zero_pad(data, length):
        extended = np.zeros(length)
        siglength = np.min([length, len(data)])
        extended[:siglength] = data[:siglength]
        return extended


    predictions = list()
    res = []
    data_name = []

    for idx, ecg_lead in enumerate(ecg_leads):
        mat_val = zero_pad(Detectors(300).hamilton_detector(mat_val),60)
        data = mat_val - np.mean(mat_val) / np.std(mat_val)
        # load the model
        model = torch.load(FILE)    
        model.eval()
        output = model(data.float())
        res.append(output.cpu().numpy().tolist())
        data_name.append(ecg_names[idx])

        if ((idx+1) % 100) == 0:
            print(str(idx+1) + "\t Dateien wurden verarbeitet.")

    res = [i[0] for i in res]
    print(f'res: {res}')
    pred = torch.tensor(res)
    print(f'pred: {pred}')
    max_value, indices = pred.topk(1, dim=1, largest=True, sorted=True)
    result = indices.tolist()
    print(np.ravel(result).tolist())
    result = np.ravel(result).tolist()

    for i in result:
        if i == 0:
            predictions.append(('A'))
        if i == 1:
            predictions.append(('N'))
        if i == 2:
            predictions.append(('O'))
        if i == 3:
            predictions.append(('~'))
    print(predictions)

    predictions = list(zip(data_name, predictions))
            
            
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
