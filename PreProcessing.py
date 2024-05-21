import os
from colorama import Fore, Back, Style
import numpy as np
import pandas as pd
import random
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from scipy.special import softmax


def File_Exist(filename):
    for folder_name, _, file_list in os.walk(r'.\file'):
        if filename in file_list:
            print(Back.GREEN + f"File '{filename}' found in folder: {folder_name}"+Style.RESET_ALL)
            return True
        else:
            print(f"File '{filename}' not found in the project.")
            return False
def PreprocessRadFeatures(radPath,clinicalPath):
    rad_features = pd.read_csv(radPath)
    cols_to_drop = [col for col in rad_features.columns if 'diagnostic' in col]
    cols_to_drop.append("Mask")
    rad_features = rad_features.drop(columns=cols_to_drop)
    rad_features["Image"] = rad_features["Image"].str[35:42]

    clinical_feature = pd.read_csv(clinicalPath)
    clinical_feature['target'] = (clinical_feature['TTP'] > 14).astype(float)

    rad_features['target'] = (clinical_feature['TTP'] > 14).astype(float)

    mat = rad_features.drop(columns='target').corr(method='spearman')

    cutoff_corr = 0.75
    high_correlation_pairs = []
    for i in range(len(mat.columns)):
        for j in range(i + 1, len(mat.columns)):
            if abs(mat.iloc[i, j]) > cutoff_corr:
                v1 = mat.columns[i]
                v2 = mat.columns[j]
                correlation_value = mat.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    pair_data = pd.DataFrame(high_correlation_pairs, columns=['feature1','feature2','correlation value'])
    drop_list = []
    for row in pair_data.iterrows():
        if random.random() > 0.5:
            drop_list.append(row[1][0])
        else:
            drop_list.append(row[1][1])
    drop_list = set(drop_list)
    rad_features = rad_features.drop(columns=drop_list)
    rad_features = rad_features.set_index('Image', drop=True).rename_axis(None)
    rad_features = rad_features.dropna(axis=0)
    print('Finished Preprocess of ', radPath)
    return rad_features
def PreprocessClinicalFeatures(csvPath):
    clinical_features = pd.read_csv(csvPath)
    clinical_features['target'] = (clinical_features['TTP'] > 14).astype(float)
    clinical_features = clinical_features.drop(columns=['age','AFP'])
    for feature in clinical_features:
        new_feature_name = feature.replace(" ","_")
        col = {feature:new_feature_name}
        clinical_features = clinical_features.rename(columns = col)
    clinical_features = clinical_features.rename(columns={'PS_bclc_0_0_1-2_1_3-4_3':'PS_bclc'})
    clinical_features = EncodeDF(clinical_features)
    clinical_features = clinical_features.drop(columns = 'Tr_Size')
    clinical_features = clinical_features.dropna(axis = 0)
    ID = clinical_features['TCIA_ID']
    clinical_features = clinical_features.drop(columns=['TCIA_ID'])
    corr = clinical_features.drop(columns=['TTP','target']).corr(method='spearman')
    cutoff_corr = 0.75
    high_correlation_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > cutoff_corr:
                v1 = corr.columns[i]
                v2 = corr.columns[j]
                correlation_value = corr.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    pair_data = pd.DataFrame(high_correlation_pairs, columns=['feature1', 'feature2', 'correlation value'])
    drop_list = []
    for row in pair_data.iterrows():
            drop_list.append(row[1][0])
    drop_list = set(drop_list)
    clinical_features = clinical_features.drop(columns=drop_list)
    cph = CoxPHFitter()
    for feature in clinical_features:
        if (feature != 'TTP') & (feature != 'target'):
            cph.fit(clinical_features,'TTP','target',formula=feature)
            # cph.print_summary()
    clinical_features['ID'] = ID.values
    clinical_features = clinical_features.rename(columns = {'target':'target_lesion'})
    return clinical_features
def EncodeDF(df):
    elab_features = []
    for feature in df:
        if df[feature].unique().size <= 10:
            # print(feature)
            elab_features.append(feature)
            values = df[feature].unique()
            try:
                values = np.sort(values)
            except:
                print(Back.RED+'Array not sorted for feature:',feature,Style.RESET_ALL)
            replace = range(0,values.size)
            dict_replace = dict(zip(values,replace))
            df[feature] = df[feature].replace(dict_replace)
    return df
def class_features(df):
    cat_feat = []
    num_feat = []
    for feature in df:
        if df[feature].unique().size <= 10:
            cat_feat.append(feature)
        else:
            num_feat.append(feature)
    return cat_feat,num_feat
def ScaleDF(df, features = None):
    scaler = StandardScaler()
    if features is None:
        df = scaler.fit_transform(df)
    else:
        df[features] = scaler.fit_transform(df[features])
    return df
def IDMerge(df1,df2):
    id_c = df1['ID']
    id_r = df2['ID']
    ids = set(id_c).intersection(id_r)
    df1 = df1[df1['ID'].isin(ids)]
    df2 = df2[df2['ID'].isin(ids)]
    return df1,df2
def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    plt.show()
def LassoAnalysis(df,y=None):
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    if y is None:
        y = df.iloc[:, -1]
    print(Style.BRIGHT + "\033[4m" + "Lasso Logistic Regression Analysis" + Style.RESET_ALL)
    x = df.iloc[:, 1:-1]
    model_lasso = LassoCV(alphas=np.linspace(2,0,50), cv=10, max_iter=100000, random_state=0).fit(x, y)
    coef = pd.Series(model_lasso.coef_, index=x.columns)
    coef_pd = pd.DataFrame({"coef" : coef})
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
    for el in coef_pd.iterrows():
        if el[1][0]>0:
            print(Fore.CYAN+str('%.4f' % el[1][0])+ Style.RESET_ALL + ' * ' + Fore.YELLOW + str(el[0])+ Style.RESET_ALL)
    print(Style.BRIGHT +"Score of The Lasso Model is "+"\033[4m" +str('%.4f' % model_lasso.score(x,y))+Style.RESET_ALL)
    return coef_pd
def LassoCoxAnalysis(x,y,plot):
    print( Style.BRIGHT + "\033[4m" + "Lasso CoxPH Model Analysis" + Style.RESET_ALL)
    coxnet_pipe = make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=10000))
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(x, y)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    gcv = GridSearchCV(coxnet_pipe,
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=1,
    ).fit(x, y)
    cv_results = pd.DataFrame(gcv.cv_results_)
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    if plot:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(alphas, mean)
        ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
        ax.set_xscale("log")
        ax.set_ylabel("concordance index")
        ax.set_xlabel("alpha")
        ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
        ax.axhline(0.5, color="grey", linestyle="--")
        plt.show()
    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(best_model.coef_, index=x.columns, columns=["coefficient"])
    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(Fore.GREEN+Style.BRIGHT+f"Number of non-zero coefficients: {non_zero}"+Style.RESET_ALL)
    print('RAD_SIGNATURE: ')
    formula = list()
    for index,row in best_coefs.iterrows():
        if row['coefficient'] != 0:
            el = [row['coefficient'],str(index)]
            formula.append(tuple(el))
            print(Fore.CYAN+str('%.4f'%row['coefficient'])+Style.RESET_ALL+' * '+Fore.YELLOW+str(index)+Style.RESET_ALL)
    if plot:
        non_zero_coefs = best_coefs.query("coefficient != 0")
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index
        _, ax = plt.subplots(figsize=(6, 8))
        non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
        ax.set_xlabel("coefficient")
        ax.grid(True)
        plt.show()
    return formula


if __name__ == '__main__':
    file_csv = "LesionFeatures.csv"
    if not File_Exist(file_csv):
        LesionRadPath = r"C:\Users\marvu\Desktop\GitHub\RadTACE\file\RadFeatures_Lesion.csv"
        ClinicalPath = r"C:\Users\marvu\Desktop\GitHub\RadTACE\file\clinical_data.csv"
        LesionRadFeatures = PreprocessRadFeatures(LesionRadPath,ClinicalPath)
        LesionRadFeatures = LesionRadFeatures.rename(columns=lambda x: str(x) + '_lesion')
        dest_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\" + file_csv
        LesionRadFeatures.to_csv(dest_path)
    else:
        dest_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\" + file_csv
        RadFeatures = pd.read_csv(dest_path)
        _ , num_cols = class_features(RadFeatures)
        num_cols = num_cols[1:]
        scaler = StandardScaler().fit(RadFeatures[num_cols])
        RadFeatures[num_cols] = scaler.transform(RadFeatures[num_cols])
        RadFeatures = RadFeatures.rename(columns={'Unnamed: 0':'ID'})
    ClinicalFeatures = PreprocessClinicalFeatures(r"C:\Users\marvu\Desktop\GitHub\RadTACE\file\clinical_data.csv")
    RadFeatures, ClinicalFeatures = IDMerge(RadFeatures, ClinicalFeatures)

    # Feature selection for radiomics features with Lasso with Logistic Regressor
    # TTP = ClinicalFeatures['TTP']
    # RadFeatures['TTP'] = TTP
    y = ClinicalFeatures['TTP']
    coef = LassoAnalysis(RadFeatures,y)

    # Feature selection for radiomics features with Lasso Using CoxPH model

    t = ClinicalFeatures['TTP'].to_numpy()
    target = ClinicalFeatures['Censored_0_progressed_1'].to_numpy()
    target = target > 0
    y = np.array(list(zip(target, t)), dtype=[('target', target.dtype), ('T', t.dtype)])
    x = RadFeatures.drop(columns = ['ID','target_lesion'])
    formula_cox = LassoCoxAnalysis(x,y,True)
    a = 0

