import os
from colorama import Fore, Back, Style
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sksurv.linear_model import CoxnetSurvivalAnalysis
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def PreprocessRadFeatures(radPath,clinicalPath, thrs):
    rad_features = pd.read_csv(radPath)
    rad_features = rad_features.dropna()
    cols_to_drop = [col for col in rad_features.columns if 'diagnostic' in col]
    cols_to_drop.append("Mask")
    rad_features = rad_features.drop(columns=cols_to_drop)
    rad_features["Image"] = rad_features["Image"].str[35:42]

    clinical_feature = pd.read_csv(clinicalPath)
    clinical_feature['target'] = (clinical_feature['TTP'] > 14).astype(float)

    rad_features['target'] = (clinical_feature['TTP'] > 14).astype(float)

    mat = rad_features.drop(columns='target').corr(method='pearson')
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
        print(i)
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
    print('Features maintained: ', len(rad_features.columns))
    return rad_features
def PreprocessClinicalFeatures(csvPath):
    clinical_features = pd.read_csv(csvPath)
    clinical_features['target'] = (clinical_features['TTP'] > 14).astype(float)
    for feature in clinical_features:
        new_feature_name = feature.replace(" ","_")
        col = {feature:new_feature_name}
        clinical_features = clinical_features.rename(columns = col)
    clinical_features = clinical_features.drop(columns=['age', 'AFP'])
    clinical_features = clinical_features.rename(columns={'PS_bclc_0_0_1-2_1_3-4_3':'PS_bclc'})
    clinical_features = EncodeDF(clinical_features)
    clinical_features = clinical_features.drop(columns = 'Tr_Size')
    clinical_features = clinical_features.dropna(axis = 0)
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
def LassoAnalysis(x,y):
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    alphas = np.linspace(0.0001, 1, 1000)
    find_alpha = LassoCV(cv = 10, random_state = seed,
                         max_iter = 10000,
                         verbose = 0,
                         alphas=alphas).fit(x,y)
    lasso = Lasso(alpha=find_alpha.alpha_).fit(x,y)
    coef = list(zip(lasso.coef_, x))
    features = []
    coeff = []
    count = 0
    for value, item in coef:
        if value != 0:
            count += 1
    print(Fore.GREEN + Style.BRIGHT + f"Number of non-zero coefficients: {count}" + Style.RESET_ALL)
    print(Fore.RED + Style.BRIGHT + f"Rad_Score: " + Style.RESET_ALL)
    print(Fore.CYAN + str('%.4f' % lasso.intercept_) + Style.RESET_ALL + ' + ')
    for value, item in coef:
        if value != 0:
            print(Fore.CYAN + str('%.4f' % value) + Style.RESET_ALL + ' * ' + Fore.YELLOW + str(item) + Style.RESET_ALL)
            features.append(item)
            coeff.append(value)
    features.append('intercept')
    coeff.append(lasso.intercept_)
    df = pd.DataFrame({"feature":features,"coeff":coeff})
    f = x[df['feature'][:-1]]
    c = df['coeff'][:-1].values
    rad_score = f * c
    rad_score['sum'] = rad_score.sum(axis=1)
    rad_score['score'] = rad_score['sum'] + df['coeff'].values[-1]
    x['target']=y
    x['rad_score'] = rad_score['score']
    return df, x
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
    features = []
    coeff = []
    for index, row in coefficients.iterrows():
        if row['coefficient'] != 0:
            print(Fore.CYAN + str('%.4f' % row['coefficient'] + Style.RESET_ALL + ' * ' + Fore.YELLOW + row['feature'] + Style.RESET_ALL))
            features.append(row['feature'])
            coeff.append(row['coefficient'])
    print('\033[1;30;47m' + '-' * 50 + '\033[0m')
    df = pd.DataFrame({"feature": features, "coeff": coeff})
    return df

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
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    log_reg_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    log_reg_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"                        Accuracy    AUC")
    print(f"Logistic Regression --> {log_reg_accuracy.mean():.4f}      {log_reg_auc.mean():.4f}")
def svm(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    svm_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    svm_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"SVM                 --> {svm_accuracy.mean():.4f}      {svm_auc.mean():.4f}")
def rf(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    grid = GridSearchCV(RandomForestClassifier(random_state=1969), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    rf_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    rf_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"Random Forest       --> {rf_accuracy.mean():.4f}      {rf_auc.mean():.4f}")
def xgb(df):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    param_grid = {'n_estimators': [50, 100, 200]}
    grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    xgb_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    xgb_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"XGBoost             --> {xgb_accuracy.mean():.4f}      {xgb_auc.mean():.4f}")
def mlp(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1969)
    x = df[[name for name in df if 'target' not in name]]
    y = df['target']
    param_grid = {'hidden_layer_sizes': [(32,), (32,16), (64,)]}
    grid = GridSearchCV(MLPClassifier(max_iter=10000, random_state=1969), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    mlp_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    mlp_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"MLP                 --> {mlp_accuracy.mean():.4f}      {mlp_auc.mean():.4f}")

if __name__ == '__main__':
    seed = 1969
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category= ConvergenceWarning)

    thr = 0.85
    thr_s = f"{int(thr * 100):03d}"
    file = "feature_"+thr_s+".csv"
    clinical_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\clinical_data.csv"
    if file not in os.listdir("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\"):
        radiomics_csv = "radiomics_features.csv"
        print('Preprocessing of file ',radiomics_csv)
        full_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\" + radiomics_csv
        RadFeatures = pd.read_csv(full_path)
        RadFeatures = PreprocessRadFeatures(full_path,clinical_path, thr)
        RadFeatures.to_csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\feature_"+thr_s+".csv")
        RadFeatures = RadFeatures.reset_index()
        RadFeatures = RadFeatures.rename(columns={"index": "ID"})
    else:
        print("Read file ",file)
        RadFeatures = pd.read_csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\feature_"+thr_s+".csv")
        RadFeatures = RadFeatures.rename(columns={"Unnamed: 0": "ID"})

    _ , num_cols = class_features(RadFeatures)
    num_cols = num_cols[1:]
    scaler = StandardScaler().fit(RadFeatures[num_cols])
    RadFeatures[num_cols] = scaler.transform(RadFeatures[num_cols])
    ClinicalFeatures = PreprocessClinicalFeatures(clinical_path)
    RadFeatures, ClinicalFeatures = IDMerge(RadFeatures,ClinicalFeatures)

    # Divided the dataset in train and test

    target = RadFeatures['target'].values
    features = RadFeatures.drop(columns=['ID','target'])
    x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=seed, )


    # Computing LASSO methods and Calculate Radiomics Score with 10f cv
    # and calculate a Rad_Score for every patient
    lasso, data_score = LassoAnalysis(x_train, y_train)
    # Find the best threshold
    data = data_score[['target','rad_score']]
    data = data.sort_values(by = 'rad_score')
    fpr, tpr, thresholds = roc_curve(data['target'],data['rad_score'])
    youden_index = tpr + (1-fpr) -1
    optimal_threshold = thresholds[np.argmax(youden_index)]
    # Plot ROC Curve
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.10])
    # plt.xlabel('Tasso di falsi positivi (False Positive Rate)')
    # plt.ylabel('Tasso di veri positivi (True Positive Rate)')
    # plt.title('Curva ROC')
    # plt.legend(loc="lower right")
    # plt.show()
    y_pred = (data['rad_score'] >= optimal_threshold).astype(int)
    accuracy = np.mean(y_pred == data['target'])
    print(0)