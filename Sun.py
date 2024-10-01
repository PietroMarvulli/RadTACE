import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torchvision import models, transforms
from PIL import Image
from radiomics import featureextractor
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


def load_image(nifti_path):
    image = sitk.ReadImage(nifti_path)
    new_spacing = [1, 1, 1]
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampled_image = resampler.Execute(image)
    return resampled_image

def load_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    new_spacing = [1, 1, 1]
    original_spacing = mask.GetSpacing()
    original_size = mask.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(mask.GetDirection())
    resampler.SetOutputOrigin(mask.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampled_mask = resampler.Execute(mask)
    mask = sitk.GetArrayFromImage(resampled_mask)
    mask = np.where(mask == 2, 1, 0)
    filtered_mask = sitk.GetImageFromArray(mask)
    filtered_mask.CopyInformation(resampled_mask)
    return filtered_mask

def extract_radiomic(nifti_path, mask_path):
    config_path = "file\\Sun\\pyrad_config_Sun.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
    features = extractor.execute(nifti_path, mask_path)
    features = {k: v for k, v in features.items() if "diagnostics" not in k}
    features = pd.DataFrame(features, index = [0])
    return features

def select_lts(ct_image, mask_image):
    max_area = 0
    best_slice_idx = 0
    ct_image = sitk.GetArrayFromImage(ct_image)
    mask_image = sitk.GetArrayFromImage(mask_image)
    if (ct_image.shape[0] <= mask_image.shape[0]):
        dim = ct_image.shape[0]-2
    else:
        dim = mask_image.shape[0]
    for i in range(dim):
        slice_mask = mask_image[i]
        tumor_area = np.sum(slice_mask)
        if tumor_area > max_area:
            max_area = tumor_area
            best_slice_idx = i
    return best_slice_idx

def extract_patches(image,mask):
    window_size = 256
    # mask = sitk.GetArrayFromImage(mask)
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return image
    # Trova il centroide del tumore
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    half_window = window_size // 2
    start_x = center_x - half_window
    start_y = center_y - half_window
    end_x = center_x + half_window
    end_y = center_y + half_window

    img_h, img_w = image.shape
    if start_x < 0:
        start_x = 0
        end_x = window_size
    if start_y < 0:
        start_y = 0
        end_y = window_size
    if end_x > img_w:
        end_x = img_w
        start_x = img_w - window_size
    if end_y > img_h:
        end_y = img_h
        start_y = img_h - window_size

    window_image = image[start_y:end_y, start_x:end_x]
    return  window_image

def ch3_patch(ct_image, mask_image, slice_idx):
    window_size = 256
    ct_image = sitk.GetArrayFromImage(ct_image)
    mask_image = sitk.GetArrayFromImage(mask_image)
    slice_prev = ct_image[max(slice_idx - 1, 0), :, :]
    slice_curr = ct_image[slice_idx, :, :]
    slice_next = ct_image[min(slice_idx + 1, ct_image.shape[0] - 1), :, :]

    mask_curr = mask_image[slice_idx, :, :]

    patch_prev = extract_patches(slice_prev, mask_curr)
    patch_curr = extract_patches(slice_curr, mask_curr)
    patch_next = extract_patches(slice_next, mask_curr)

    patch_prev = cv2.normalize(patch_prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    patch_curr = cv2.normalize(patch_curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    patch_next = cv2.normalize(patch_next, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    patch_3channel = np.stack([patch_prev, patch_curr, patch_next], axis=-1)
    patch_3channel = Image.fromarray(np.uint8(patch_3channel))
    return patch_3channel

def preprocess_patch(patch):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_patch = preprocess(patch).unsqueeze(0)
    return processed_patch

def deep_image(image,mask):
    index = select_lts(image, mask)
    patch = ch3_patch(image,mask,index)
    patch_proc = preprocess_patch(patch)
    return index,patch_proc

def extract_deep_features(patch_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18 = models.resnet18(pretrained=True, progress = True)
    modules = list(resnet18.children())[:-1]
    model = torch.nn.Sequential(*modules)
    model.to(device)
    model.eval()
    patch_tensor = patch_tensor.to(device)
    with torch.no_grad():
        features = model(patch_tensor).squeeze().flatten().cpu().numpy()
    features = pd.DataFrame(features.reshape(1,-1))
    return features

def clinical_analysis(clinical_data):
    global results_ua, results_ma, final_significant_df
    results = []
    features_to_keep = []
    for col in clinical_data.columns:
        if col not in ['target','ID','TTP']:
            num_unique_values = clinical_data[col].nunique()
            if num_unique_values > 10:
                group1 = clinical_data[clinical_data['target'] == 0][col]
                group2 = clinical_data[clinical_data['target'] == 1][col]
                t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')
                results.append({
                    'feature': col,
                    'type': 'Continuous',
                    'test': 'T-test',
                    'p-value': p_val
                })

            else:  # Variabile categorica
                contingency_table = pd.crosstab(clinical_data[col], clinical_data['target'])
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                results.append({
                    'feature': col,
                    'type': 'Categorical',
                    'test': 'Chi-squared',
                    'p-value': p_val
                })
            if p_val <= 0.1:
                features_to_keep.append(col)
            results_ua = pd.DataFrame(results)

    if len(features_to_keep) > 0:
        X = clinical_data[features_to_keep]
        y = clinical_data['target']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logit_model = sm.Logit(y, X)
        results_ma = logit_model.fit(disp=0)
        p_values = results_ma.pvalues
        p_values.index = features_to_keep
        significant_features = p_values[p_values <= 0.1].index
        final_significant_df = clinical_data[significant_features]
        final_significant_df = final_significant_df.copy()
        final_significant_df['target'] = clinical_data.iloc[:,-1]

    return results_ua, results_ma, final_significant_df

def correlation_analysis(data, thrs):
    mat = data.corr().abs()
    high_correlation_pairs = []
    for i in tqdm(range(len(mat.columns)), ascii=True, desc="Processing correlations"):
        for j in range(i + 1, len(mat.columns)):
            if abs(mat.iloc[i, j]) > thrs:
                v1 = mat.columns[i]
                v2 = mat.columns[j]
                correlation_value = mat.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    pair_data = pd.DataFrame(high_correlation_pairs, columns=['feature1', 'feature2', 'correlation value'])
    drop_list = []
    for index, row in tqdm(pair_data.iterrows(), total=pair_data.shape[0], ascii=True, desc=f"Evaluating features "):
        feature1 = row['feature1']
        feature2 = row['feature2']
        var1 = data[feature1].var()
        var2 = data[feature2].var()
        if var1 < var2:
            ftr = feature1
        else:
            ftr = feature2

        if ftr not in drop_list:
            drop_list.append(ftr)

    data = data.drop(columns=drop_list)
    return data

def lasso_analysis(data, target, cv):
    warnings.simplefilter("ignore", ConvergenceWarning)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    y = target.values
    lasso_cv = LassoCV(cv=10, random_state= 42).fit(data_scaled, y)
    a = lasso_cv.alpha_
    lasso_single = Lasso(alpha=a).fit(data_scaled, y)
    selected_features = pd.Series(lasso_single.coef_, index=data.columns)
    selected_features = selected_features[selected_features != 0]
    filtered_data = data[selected_features.index]
    return filtered_data

def lr(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    features = [name for name in df if 'target' not in name]
    x = df[features]
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    log_reg_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    log_reg_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')

    print(f"Logistic Regression --> {log_reg_accuracy.mean():.4f}      {log_reg_auc.mean():.4f}")

if __name__ == "__main__":
    image_mask = pd.read_csv("utils\\image-mask-final.csv")
    clinical_data = pd.read_csv("utils\\clinical-data-final.csv")

    results_ua, results_ma, clinical_data = clinical_analysis(clinical_data)

    # Radiomics = pd.DataFrame()
    # Deeps = pd.DataFrame()
    # for row, item in tqdm(image_mask.iterrows(), total=len(image_mask)):
    #     image_path = item[0].split(';')[0]
    #     mask_path = item[0].split(';')[1]
    #     image = load_image(image_path)
    #     mask = load_mask(mask_path)
    #     radiomics_features = extract_radiomic(image_path,mask_path)
    #     Radiomics = pd.concat([Radiomics, radiomics_features], ignore_index=True)
    #     array_image = sitk.GetArrayFromImage(image)
    #     array_mask = sitk.GetArrayFromImage(mask)
    #     index, patch = deep_image(image,mask)
    #     # print(array_mask.shape, array_mask.shape, index)
    #     deep_features = extract_deep_features(patch)
    #     Deeps = pd.concat([Deeps, deep_features], ignore_index=True)

    target = clinical_data['target'].values
    radiomics_features = pd.read_csv("file/Sun/Radiomics.csv")
    deep_features = pd.read_csv("file/Sun/Deeps.csv")

    deep_features = correlation_analysis(deep_features, 0.9)
    radiomics_features = correlation_analysis(radiomics_features, 0.9)
    deep_lasso = lasso_analysis(deep_features, clinical_data['target'],5)
    radiomics_lasso = lasso_analysis(radiomics_features, clinical_data['target'],5)
    data = pd.concat([clinical_data, deep_lasso, radiomics_lasso], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data.iloc[:,:] = scaled_data
    print("                        Accuracy    AUC")
    lr(data,target)