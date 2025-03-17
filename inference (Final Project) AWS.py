#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import pickle
import xgboost as xgb
import numpy as np
import json

def model_fn(model_dir):
    # This is to load the XGBoost model from the saved file
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data)
    else:
        raise ValueError("Content type not supported")

def predict_fn(input_data, model):
    return model.predict(input_data)


# In[ ]:




