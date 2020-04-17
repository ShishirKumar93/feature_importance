#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:19:52 2020

@author: MrMndFkr
"""

import pandas as pd
import numpy as np
from typing import Mapping
from matplotlib import pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone
import shap

def spearman_corr(df: pd.DataFrame, target:str) -> np.array:
    "Returns an array of spearman's rank correlation for each of the features calculated w.r.t. the input column"
    df = df.copy()
    corr = {}
    cols = [col for col in df.columns if col != target]
    rank_target = df[target].rank(axis=0, method = 'min')
    for col in cols:
        rank_feat = df[col].rank(axis=0, method='average') ## average out the rank for points with same rank
        cov = rank_feat.cov(rank_target)
        std_target, std_feat = rank_target.std(), rank_feat.std()
        corr_feat = cov / std_target / std_feat
        corr[col] = abs(corr_feat)
    return dict(sorted(corr.items(), key=lambda x: x[1], reverse=True))

def plot_feat_imp(importances:dict, title:str, show_values=False) -> plt:
    "Returns a horizontal bar plot of feature importances"
    fig, ax = plt.subplots(figsize=(15,10))
    ax.barh(np.array([val for val in importances.keys()]), 
            np.array([val for val in importances.values()]), color='lightblue') # Horizontal Bar Plot
    for s in ['top','bottom','left','right']:
        ax.spines[s].set_visible(False) # Remove axes splines
    ax.xaxis.set_ticks_position('none') # Remove x,y Ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad=5) # Add padding between axes and labels
    ax.yaxis.set_tick_params(pad=10)
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2) # Add x,y gridlines
    ax.invert_yaxis()# Show top values 
    ax.set_title(title, loc='left', pad=-5, fontweight='bold', fontsize=15)# Add Plot Title
    if show_values:
        for i in ax.patches: # Add annotation to bars
            if i.get_width() < 0: a = i.get_width()-0.035
            else: a = i.get_width()+0.005
            ax.text(a, i.get_y()+0.6, str(round((i.get_width()), 2)), fontsize=10, fontweight='bold', color='grey')
    fig.text(0.9, 0.15, '@skumar18', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.5) # Add Text watermark
    return plt

def mRMR(df:pd.DataFrame, target:str) -> dict:
    """
    Returns a dictionary of features with their mRMR importances calcuated using the Spearman's correlation metric
    df: input data 
    target: name of target column
    """
    mrmr_values = dict()
    selected_feat = []
    candidates = [col for col in df.columns if col != target]
    corr_w_target = spearman_corr(df=df, target=target)
    first_feat = list(corr_w_target.keys())[0]
    selected_feat.append(first_feat)
    rest_feat = [col for col in candidates if col not in selected_feat]
    mrmr_values[first_feat] = corr_w_target[first_feat]
    i = 0
    print(f'iter {i} selected feature: {first_feat} with mRMR = {mrmr_values[first_feat]:.3f}')
    while len(rest_feat) > 0:
        i += 1
        feat_mrmr = []
        for fi in rest_feat:
            redundancy = 0
            relevance = corr_w_target[fi]
            for fj in selected_feat:
                dict_red = spearman_corr(df=df.drop(columns=target), target=fj)
                redundancy += dict_red[fi]
            redundancy = redundancy / len(selected_feat)
            mrmr = relevance - redundancy
            feat_mrmr.append(mrmr)
        selected_feature = rest_feat[np.argmax(feat_mrmr)]
        mrmr_values[selected_feature] = np.max(feat_mrmr)
        print(f'iter {i} selected feature: {selected_feature} with mRMR = {mrmr_values[selected_feature]:.3f}')
        rest_feat = [col for col in rest_feat if col != selected_feature]
        selected_feat.append(selected_feature)
    return mrmr_values, selected_feat

def dropcol_importances(rf, df:pd.DataFrame, target:str) -> dict:
    """
    Returns a dictionary of feature : importance scores for given random forest model, initialised with oob_score=True
    """
    feat_imp = dict()
    rf_ = clone(rf)
    rf_.random_state = 999
    X_train = df.drop(columns=target)
    Y_train = df[target]
    rf_.fit(X_train, Y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(columns=col)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, Y_train)
        o = rf_.oob_score_
        feat_imp[col] = baseline - o
    return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

def permutation_importance(model, val:pd.DataFrame, target:str, metric) -> dict:
    """
    Returns a dictionary of feature : importance scores for given model, metric and validation dataset
    """
    val = val.copy()
    baseline = metric(val[target], model.predict(val.drop(columns=target)))
    feat_imp = dict()
    features = [col for col in val.columns if col != target]
    for col in features:
        col_saved = val[col].copy()
        val[col] = np.random.permutation(val[col])
        new_score = metric(val[target], model.predict(val.drop(columns=target)))
        val[col] = col_saved
        feat_imp[col] = baseline - new_score
    return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

def compare_top10(model, train:pd.DataFrame, val:pd.DataFrame, feat_imp:dict, target:str, metric=log_loss) -> list:
    """
    Returns a list of log-loss values for top 1,2,3,....,10 features taken from a sorted dictionary of feature importances. 
    """
    loss_list = []
    train = train.copy()
    val = val.copy()
    for i in range(1, 11):
        model_ = clone(model)
        model_.random_state = 42
        features = [col for col in feat_imp.keys()][:i] ## select top i features
        model_.fit(train.loc[:, features], train[target])
        valid_proba = model_.predict_proba(val.loc[:, features])
        log_loss_valid = metric(val[target], valid_proba)
        loss_list.append(log_loss_valid)
    return loss_list

def get_best_model(model, train:pd.DataFrame, val:pd.DataFrame, target:str , metric=log_loss) -> list:
    """
    Returns the best model obtained by dropping features one by one, feature with minimum shap importance is dropped
    and the new validation loss is checked against the previous loss value to check if dropping the feature decreased loss.
    """
    model_ = clone(model)
    model_.random_state = 999
    X_train = train.drop(columns=target)
    Y_train = train[target]
    X_val = val.drop(columns=target)
    Y_val = val[target]
    model_.fit(X_train, Y_train)
    preds = model_.predict_proba(X_val)
    val_loss = metric(Y_val, preds)
    shap_values = shap.TreeExplainer(model_, data=X_train).shap_values(X_val)
    shap_imp = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0) ## mean shap value for each feature
    feat = [col for col in X_val.columns]
    removed_feat = []
    i = 0
    while True:
        i += 1
        remove_feat = feat[np.argmin(shap_imp)]
        removed_feat.append(remove_feat)
        selected_feat = [col for col in feat if col not in removed_feat]
        print(f'Iter {i}: dropping feature: {remove_feat}')
        model_ = clone(model)
        model_.random_state = 999
        X_train_new = X_train.drop(columns=removed_feat)
        X_val_new = X_val.drop(columns=removed_feat)
        model_.fit(X_train_new, Y_train)
        preds_new = model_.predict_proba(X_val_new)
        val_loss_new = metric(Y_val, preds_new)
        if val_loss_new <= val_loss:
            print(f'Loss in last iter: {val_loss:.3f}, loss in this iter: {val_loss_new:.3f}')
            shap_values_new = shap.TreeExplainer(model_, data=X_train_new ).shap_values(X=X_val_new, y=Y_val)
            shap_imp_new = np.sum(np.mean(np.abs(shap_values_new), axis=1), axis=0) ## mean shap value for each feature
            print(f'Loss decreased hence continuing to drop more features')
            shap_imp = shap_imp_new
            feat = [col for col in X_train_new.columns]
        else:
            removed_feat = [col for col in removed_feat if col != remove_feat]
            print(f'Loss in last iter: {val_loss:.3f}, loss in this iter: {val_loss_new:.3f}')
            print(f'Stopping iterations as loss did not decrease, dropped features = {removed_feat}')
            return model_, removed_feat
        val_loss = val_loss_new

def get_std(model, train:pd.DataFrame, val:pd.DataFrame, target:str, metric=log_loss) -> (np.array, np.array):
    """
    Returns 1 standard deviation error values for shap importance values of each feature
    """
    shap_imp = np.zeros((100,val.shape[1] - 1))
    for i in range(100):
        idx = np.random.choice(range(val.shape[0]), size=val.shape[0], replace=True) ## booststrap
        val_new = val.iloc[idx, :]
        shap_values = shap.TreeExplainer(model, data=train.drop(columns=target)).shap_values(
                                        X=val_new.drop(columns=target), y=val_new[target])
        shap_imp[i] = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    return np.std(shap_imp, axis=0) ## calculate 1 standard deviation for each feature

def plot_feat_imp_erros(importances:dict, title:str, show_values=False) -> plt:
    """
    Returns a plot of feature importances with error bars, importances is a dict of feature:(importance, error)
    """
    fig, ax = plt.subplots(figsize=(15,10))
    ax.barh(np.array([val for val in importances.keys()]), 
            np.array([val[0] for val in importances.values()]), 
            xerr=np.array([val[1] for val in importances.values()]), color='lightblue') # Horizontal Bar Plot
    for s in ['top','bottom','left','right']:
        ax.spines[s].set_visible(False) # Remove axes splines
    ax.xaxis.set_ticks_position('none') # Remove x,y Ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad=5) # Add padding between axes and labels
    ax.yaxis.set_tick_params(pad=10)
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2) # Add x,y gridlines
    ax.invert_yaxis()# Show top values 
    ax.set_title(title, loc='left', pad=-5, fontweight='bold', fontsize=15)# Add Plot Title
    if show_values:
        for i in ax.patches: # Add annotation to bars
            if i.get_width() < 0: a = i.get_width()-0.035
            else: a = i.get_width()+0.01
            ax.text(a, i.get_y()+0.6, str(round((i.get_width()), 2)), fontsize=10, fontweight='bold', color='grey')
    fig.text(0.9, 0.15, '@skumar18', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.5) # Add Text watermark
    return plt

def p_values(model, train:pd.DataFrame, val:pd.DataFrame, target:str, metric=log_loss) -> (np.array, np.array):
    """
    Returns p-value, associated array of shap baseline values and new importance scores obtained by scrambling target (Null distribution)
    """
    shap_imp = np.zeros((100,val.shape[1] - 1))
    X_train = train.drop(columns=target)
    X_val = val.drop(columns=target)
    Y_val = val[target]
    shap_values_baseline = shap.TreeExplainer(model, data=X_train).shap_values(X=X_val, y=Y_val, 
                                                                               check_additivity=False)
    shap_baseline = np.sum(np.mean(np.abs(shap_values_baseline), axis=1), axis=0)
    shap_baseline = shap_baseline / np.sum(shap_baseline) ## normalise for further comparison
    for i in range(100):
        Y_train = np.random.permutation(train[target])
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(X_train, Y_train)
        shap_values = shap.TreeExplainer(model_, data=X_train).shap_values(X=X_val, y=Y_val, check_additivity=False)
        shap_imp[i] = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0) ## recalculate and store shap
        shap_imp[i] = shap_imp[i] / np.sum(shap_imp[i]) ## Normalise for comparison
    diff = shap_baseline - shap_imp ## -ve values correspond to greater than baseline score
    p_values = np.sum(diff <= 0, axis=0) / 100 ## proportion of positive elements gives empirical p-value
    return p_values, shap_baseline, shap_imp

def get_hist(p_values:np.array, baseline_score:np.array, imp_scores:np.ndarray, features:list, n:int) -> list:
    """
    Returns a Null distribution histogram for given feature index 'n', vertical line shows the baseline importance
    """
    list_plots = []
    fig,ax = plt.subplots(figsize=(12,8))
    _ = plt.hist(imp_scores[:, n], bins='auto')
    if p_values[n] < 0.05:
        plt.title(f"Histogram of Null Distributions for significant feature: {features[n]}, Calculated p-value: {p_values[n]}")
    else:
        plt.title(f"Histogram of Null Distributions for insignificant feature: {features[n]}, Calculated p-value: {p_values[n]}")
    ax.axvline(x=shap_baseline[n], c='red')
    plt.show()
