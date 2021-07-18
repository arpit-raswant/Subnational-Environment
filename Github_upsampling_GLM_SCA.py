#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:23:42 2021

@author: jessicaleung
"""

from sklearn.utils import resample, shuffle
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import math
import statsmodels
from scipy import stats
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Read Data Set
data = pd.read_csv("data.csv", encoding='ISO-8859-1')
list_of_vars = ['Here is a list of variable names you are interested in']
#Descriptive Stat
data.describe()[list_of_vars]

#Upsampling
df_1 = data[data['Y'] == 1]
#set other classes to another dataframe
other_df = data[data['Y'] != 1]  
#upsample the minority class
df_1_upsampled = resample(df_1,random_state=0,replace=True,n_samples=len(other_df))
#concatenate the upsampled dataframe
df_1_upsampled = pd.concat([df_1_upsampled,other_df])

sub_df_y = df_1_upsampled[['Y']]
sub_df_X = df_1_upsampled.drop(columns=['Y'])
sub_df_X['Const'] = np.ones(len(sub_df_y))

# GLM
X_train, X_test, y_train, y_test = train_test_split(sub_df_X, sub_df_y, test_size=0.2, random_state=0,shuffle=True)
res = GLM(y_train, X_train,
         family=families.Binomial() ).fit(attach_wls=True, atol=1e-10, cov_type='HC1')
print(res.summary())
#Prediction Performance
#Training Confusion Matrix
y_hat = np.sign(res.predict()-0.5)
y_hat = 0.5*(y_hat + 1)
confusion_matrix(y_train, y_hat)
#Testing Confusion Matrix
y_hat_test = np.sign(res.predict(X_test) - 0.5)
y_hat_test = 0.5*(y_hat_test + 1)
confusion_matrix(y_test, y_hat_test)

# Specification Curve Analysis
#Here we define a list of variables that we are interested in the SCA, excluding the control variables
var_list = ['Here is a list of variables']
for interested_var in var_list:
    #Upsampling
    df_1 = data[data['Y'] == 1]
    #set other classes to another dataframe
    other_df = data[data['Y'] != 1]  
    #upsample the minority class
    df_1_upsampled = resample(df_1,random_state=0,replace=True,n_samples=len(other_df))
    #concatenate the upsampled dataframe
    df_1_upsampled = pd.concat([df_1_upsampled,other_df])
    
    import itertools as its
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return its.chain.from_iterable(its.combinations(s, r) for r in range(len(s)+1))
    Specs = powerset(var_list)

    #for var_name in 
    Results_df = {"pvals":[0],"coeff":[0],"conf_lower":[0], "conf_higher":[0]}
    Results_df = pd.DataFrame(Results_df)
    Control = ['Here is a list of control variables']
    for spec_id, spec in enumerate(powerset(var_list)):
        print(spec)
        spec=list(spec)
        spec.append('Y')
        [spec.append(item) for item in Control]
        if interested_var in spec:
            sca_filtered_data = df_1_upsampled[spec]
            sub_df_y = sca_filtered_data[['Y']]
            sub_df_X = sca_filtered_data.drop(columns=['Y'])
            sub_df_X['Const'] = np.ones(len(sub_df_y))
            X_train, X_test, y_train, y_test = train_test_split(sub_df_X, sub_df_y, test_size=0.2, random_state=0,shuffle=True)
            res = GLM(y_train, X_train,
                     family=families.Binomial() ).fit(attach_wls=True, atol=1e-10, cov_type='HC1')
            print(res.summary())

            pvals = res.pvalues[interested_var]
            coeff = res.params[interested_var]
            conf_lower = res.conf_int()[0][interested_var]
            conf_higher = res.conf_int()[1][interested_var]
            new_row = {"pvals":pvals, "coeff":coeff, "conf_lower":conf_lower, "conf_higher":conf_higher}
            Results_df=Results_df.append(new_row, ignore_index=True)        
    
    #Stripped line 0
    Results_df = Results_df.drop(0)
    Results_df = Results_df.reset_index()
    Sorted_results = Results_df.sort_values("coeff")
    heatmap = []
    for var in var_list:
        print(var)
        row=[]
        for spec in powerset(var_list):
            print(spec)
            if interested_var in spec:
                if (var in spec):
                    row.append(1)
                else:
                    row.append(0)
        heatmap.append(row) 
    heatmap = pd.DataFrame(np.array(heatmap))
    heatmap = heatmap[Sorted_results.index.tolist()]
    
    #plot the graph
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(16,8), gridspec_kw={'height_ratios': [2, 1]})
    
    x_pos=0
    for spec_id, lower, upper, coeff, pvalue in zip(Sorted_results.index, Sorted_results['conf_lower'], Sorted_results['conf_higher'], Sorted_results['coeff'], Sorted_results['pvals']):
        if pvalue<=0.05:
            ax1.plot( (x_pos,x_pos),(lower, upper), '-', color='orange')
            ax1.plot(x_pos, coeff, 'o', color='orange')
        else:
            ax1.plot( (x_pos,x_pos),(lower, upper), '-', color='grey')
            ax1.plot(x_pos, coeff, 'o', color='grey')
        x_pos+=1
        print(x_pos)    
    ax1.set_ylabel('Coefficient')
    ax1.plot(np.linspace(0,len(Sorted_results),num=200),np.zeros(200),'k--')
    
    ax2=plt.imshow(heatmap,cmap='gist_gray_r')
    plt.yticks(range(len(var_list)),labels=var_list)
    plt.tight_layout(pad=-3)
    plt.show()

