import os
from colorama import Fore, Back, Style
import numpy as np
import pandas as pd
import random
from lifelines import CoxPHFitter
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sksurv.linear_model import CoxnetSurvivalAnalysis
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def PreprocessRadFeatures(radPath,clinicalPath, thrs):
    rad_features = pd.read_csv(radPath)
    cols_to_drop = [col for col in rad_features.columns if 'diagnostic' in col]
    cols_to_drop.append("Mask")
    rad_features = rad_features.drop(columns=cols_to_drop)
    rad_features["Image"] = rad_features["Image"].str[35:42]

    clinical_feature = pd.read_csv(clinicalPath)
    clinical_feature['target'] = (clinical_feature['TTP'] > 14).astype(float)

    rad_features['target'] = (clinical_feature['TTP'] > 14).astype(float)

    mat = rad_features.drop(columns='target').corr(method='spearman')
    cluster_map = False
    if cluster_map:
        sns.set(style='white')
        clustermap = sns.clustermap(mat, method='average', cmap='vlag', linewidths=0.75, figsize=(200, 200))
        plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)  # Keep y-labels horizontal
        plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90)  # Rotate x-labels for readability
        clustermap.savefig('clustermap.png', format='png')
        plt.show()
    cutoff_corr = thrs
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
def PreprocessClinicalFeatures(csvPath, thrs):
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
    cutoff_corr = thrs
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
    clinical_features['ID'] = ID.values
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
def LassoAnalysis(df,y=None, plot = False):
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    if y is None:
        y = df.iloc[:, -1]
    print(Style.BRIGHT + "\033[4m" + "Lasso Logistic Regression Analysis" + Style.RESET_ALL)
    x = df.iloc[:, 1:-1]
    model = LassoCV(cv=10, max_iter=10000, random_state=1969)
    model.fit(df.iloc[:, 1:-1], y)
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(df.iloc[:, 1:-1], y)
    coef = list(zip(lasso_best.coef_, df.iloc[:, 1:-1]))
    features = []
    coeff = []
    for value, item in coef:
        if value != 0:
            print(Fore.CYAN + str('%.4f' % value) + Style.RESET_ALL + ' * ' + Fore.YELLOW + str(item) + Style.RESET_ALL)
            features.append(item)
            coeff.append(value)
    # print('R squared value', round(lasso_best.score(x, y) * 100, 2))
    # print(mean_squared_error(y, lasso_best.predict(x)))
    df = pd.DataFrame({"feature":features,"coeff":coeff})
    if plot:
        plt.semilogx(model.alphas_, model.mse_path_, ":")
        plt.plot(
            model.alphas_,
            model.mse_path_.mean(axis=-1),
            "k",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(
            model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
        )
        plt.legend()
        plt.xlabel("alphas")
        plt.ylabel("Mean square error")
        plt.title("Mean square error on each fold")
        plt.axis("tight")
        plt.show()
    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    return df
def LassoCoxAnalysis(x,y,plot = False):
    print( Style.BRIGHT + "\033[4m" + "Lasso CoxPH Model Analysis" + Style.RESET_ALL)
    coxnet_pipe = make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=0.01, max_iter=10000))
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(x, y)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=10, shuffle=True, random_state=1969)
    gcv = GridSearchCV(coxnet_pipe,
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv).fit(x, y)
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
    features = []
    coef = []
    for el in formula:
        coef.append(el[0])
        features.append(el[1])
    formula = pd.DataFrame({'feature':features,'coeff':coef})
    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    return formula
def ElasticNetAnalysis(x,y):
    print( Style.BRIGHT + "\033[4m" + "Elastic Net Analysis" + Style.RESET_ALL)
    elastic_net = ElasticNetCV(cv=10, random_state=1969)
    elastic_net.fit(x, y)
    optimal_alpha = elastic_net.alpha_
    optimal_l1_ratio = elastic_net.l1_ratio_
    elastic_net_optimal = ElasticNet(alpha=optimal_alpha, l1_ratio=optimal_l1_ratio)
    elastic_net_optimal.fit(x, y)
    coefficients = pd.DataFrame({'feature':elastic_net_optimal.feature_names_in_,'coefficient': elastic_net_optimal.coef_})
    non_zero = np.sum(abs(coefficients.iloc[:, 1]) > 0)
    print(Fore.GREEN + Style.BRIGHT + f"Number of non-zero coefficients: {non_zero}" + Style.RESET_ALL)
    for index, row in coefficients.iterrows():
        if row['coefficient'] != 0:
            print(Fore.CYAN + str('%.4f' % row['coefficient'] + Style.RESET_ALL + ' * ' + Fore.YELLOW + row['feature'] + Style.RESET_ALL))
    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    return coefficients
def UnivariateAnalysis(df):
    cph = CoxPHFitter()
    df = df.drop(columns=['ID'])
    covariates = [col for col in df.columns if col not in ['OS','Death_1_StillAliveorLostToFU_0']]
    results = pd.DataFrame()
    for feature in covariates:
        cph.fit(df[[feature,'OS','Death_1_StillAliveorLostToFU_0']], duration_col='OS', event_col='Death_1_StillAliveorLostToFU_0')
        summary = cph.summary.transpose()
        results[feature] = summary
    return results
def MultivariateAnalysis(df, group = None):
    cph = CoxPHFitter()
    df = df.drop(columns=['ID'])
    if group is None:
        cph.fit(df, duration_col='OS', event_col='Death_1_StillAliveorLostToFU_0')
        summary = cph.summary.transpose()
    else:
        group.append('Death_1_StillAliveorLostToFU_0')
        group.append('OS')
        df = df[group]
        cph.fit(df, duration_col='OS', event_col='Death_1_StillAliveorLostToFU_0')
        summary = cph.summary.transpose()
    return summary
def PrintRevelantFeatures(df,thr):
    if thr == 0.05:
        method = Style.BRIGHT+'Univariate'+Style.RESET_ALL
    else:
        method = Style.BRIGHT+'Multivariate'+Style.RESET_ALL
    if 'p' in df.index:
        ss_features = df.columns[df.loc['p']<thr]
        print('Feature selezionate con analisi '+method+':')
        for col in ss_features:
            print('\t'+col)
def Tsne2D(df):
    warnings.simplefilter("ignore", FutureWarning)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    tsne_2d = TSNE(n_components=2, random_state=1969)
    tsne_2d_results = tsne_2d.fit_transform(features)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_2d_results[:, 0], tsne_2d_results[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE 2D')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()
def Tsne3D(df):
    warnings.simplefilter("ignore", FutureWarning)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    tsne_3d = TSNE(n_components=3, random_state=1969)
    tsne_3d_results = tsne_3d.fit_transform(features)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_3d_results[:, 0], tsne_3d_results[:, 1], tsne_3d_results[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    ax.set_title('t-SNE 3D')
    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')
    ax.set_zlabel('t-SNE component 3')
    plt.show()
def lr(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    model = LogisticRegression().fit(x,y)
    log_reg_accuracy = cross_val_score(model, x, y, cv=skf, scoring= 'accuracy')
    log_reg_auc = cross_val_score(model, x, y, cv=skf, scoring='roc_auc')
    print(f"Logistic Regression Accuracy: {log_reg_accuracy.mean():.4f}")
    print(f"Logistic Regression AUC: {log_reg_auc.mean():.4f}\n")
def svm(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    model = SVC(probability=True).fit(x,y)
    svm_accuracy = cross_val_score(model, x, y, cv=skf, scoring= 'accuracy')
    svm_auc = cross_val_score(model, x, y, cv=skf, scoring='roc_auc')
    print(f"SVM Accuracy: {svm_accuracy.mean():.4f}")
    print(f"SVM AUC: {svm_auc.mean():.4f}\n")
def rf(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    rf = RandomForestClassifier(n_estimators=100, random_state=1969).fit(x,y)
    rf_accuracy = cross_val_score(rf, x, y, cv=skf, scoring='accuracy')
    rf_auc = cross_val_score(rf, x, y, cv=skf, scoring='roc_auc')
    print(f"Random Forest Accuracy: {rf_accuracy.mean():.4f}")
    print(f"Random Forest AUC: {rf_auc.mean():.4f}\n")
def mlp(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000, random_state=1969).fit(x,y)
    mlp_accuracy = cross_val_score(mlp, x, y, cv=skf, scoring='accuracy')
    mlp_auc = cross_val_score(mlp, x, y, cv=skf, scoring='roc_auc')
    print(f"MLP Accuracy: {mlp_accuracy.mean():.4f}")
    print(f"MLP AUC: {mlp_auc.mean():.4f}\n")



if __name__ == '__main__':
    file = "feature_060.csv"
    clinical_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\clinical_data.csv"
    if file not in os.listdir("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file"):
        radiomics_csv = "RadFeatures_Lesion.csv"
        full_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\" + radiomics_csv
        RadFeatures = pd.read_csv(full_path)
        RadFeatures = PreprocessRadFeatures(full_path,clinical_path, 0.60)
        RadFeatures.to_csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\feature_060.csv")
        RadFeatures = RadFeatures.reset_index()
        RadFeatures = RadFeatures.rename(columns={"index": "ID"})
    else:
        RadFeatures = pd.read_csv(r"C:\Users\marvu\Desktop\GitHub\RadTACE\file\feature_060.csv")
        RadFeatures = RadFeatures.rename(columns={"Unnamed: 0": "ID"})

    _ , num_cols = class_features(RadFeatures)
    num_cols = num_cols[1:]
    scaler = StandardScaler().fit(RadFeatures[num_cols])
    RadFeatures[num_cols] = scaler.transform(RadFeatures[num_cols])
    ClinicalFeatures = PreprocessClinicalFeatures(clinical_path, 0.7)
    RadFeatures, ClinicalFeatures = IDMerge(RadFeatures, ClinicalFeatures)
    #
    # # Feature selection for radiomics features with Lasso with Logistic Regressor
    y = ClinicalFeatures['target']
    lasso = LassoAnalysis(RadFeatures,y)

    #
    # # Feature selection for radiomics features with Lasso Using CoxPH model
    t = ClinicalFeatures['TTP'].to_numpy()
    target = ClinicalFeatures['target'].to_numpy()
    target = target > 0
    y = np.array(list(zip(target, t)), dtype=[('target', target.dtype), ('T', t.dtype)])
    x = RadFeatures.drop(columns = ['ID','target'])

    lassocox = LassoCoxAnalysis(x,y)
    #
    # # Feature selection for radiomics features with Elastic Net (have Lasso component and Ridge component)
    x = RadFeatures.drop(columns=['ID', 'target'])
    y = ClinicalFeatures['target']
    elasticnet = ElasticNetAnalysis(x,y)
    #


    feature_set1 = set(lasso['feature'])
    feature_set2 = set(lassocox['feature'])
    feature_set3 = set(elasticnet['feature'])

    feature_list = feature_set1.intersection(feature_set2).intersection(feature_set3)
    feature_list.add('target')
    data = RadFeatures[feature_list]

    #Univariate and Multivariate anlysis are done with time = 'OS' and event = 'Death'
    univariate = UnivariateAnalysis(ClinicalFeatures)
    ff = []
    for el in univariate:
        if univariate[el].p <= 0.1:
            ff.append(el)
    multivariate = MultivariateAnalysis(ClinicalFeatures, ff)
    cf = []
    for el in multivariate:
        if multivariate[el].p <= 0.5:
            cf.append(el)
    cf.append('target')
    data_clinical = ClinicalFeatures[cf]

    print(Fore.GREEN + Style.BRIGHT + f"Print info About Radiomics and Clinical Data: " + Style.RESET_ALL)
    rad_cols = [el for el in data.columns.tolist() if el != 'target']
    clin_cols = [el for el in data_clinical.columns.tolist() if el != 'target']
    max_length = max(len(rad_cols), len(clin_cols))
    rad_cols.extend(['---'] * (max_length - len(rad_cols)))
    clin_cols.extend(['---'] * (max_length - len(clin_cols)))
    summary = pd.DataFrame({'Radiomics':rad_cols,'Clinical':clin_cols})
    print(summary)

    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    lr(data)
    svm(data)
    rf(data)
    mlp(data)

    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    lr(data_clinical)
    svm(data_clinical)
    rf(data_clinical)
    mlp(data_clinical)

    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    data_clinical = data_clinical.reset_index().drop(columns={'index'})
    data_combined = result = pd.concat([data.set_index('target'), data_clinical.set_index('target')], axis=1, join='outer').reset_index()

    lr(data_combined)
    svm(data_combined)
    rf(data_combined)
    mlp(data_combined)



