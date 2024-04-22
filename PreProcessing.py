import os
import pandas as pd
import numpy as np

rad_features = pd.read_csv(r"file/RadFeatures_Lesion.csv")
cols_to_drop = [col for col in rad_features.columns if 'diagnostic' in col]
cols_to_drop.append("Mask")
rad_features = rad_features.drop(columns=cols_to_drop)
rad_features["Image"] = rad_features["Image"].str[35:42]
clinical_feature = pd.read_csv(r"file/clinical_data.csv")
clinical_feature['target'] = (clinical_feature['TTP'] > 14).astype(float)
rad_features['target'] = (clinical_feature['TTP'] > 14).astype(float)
# Calculate Pearson Correlation for radiomics features
mat = rad_features.drop(columns='target').corr()
mat_np = np.array(mat)
mat_np[np.tril_indices_from(mat_np)] = 0
np.fill_diagonal(mat_np,1)
mat = pd.DataFrame(mat_np, index = mat.index.tolist(),columns=mat.columns.tolist())
# plt.subplots(figsize=(100, 80))
# sns.heatmap(mat,cmap='coolwarm', fmt=".2f", xticklabels=False, yticklabels=False)
# plt.title("Correlation Matrix")
# plt.show()
cutoff_corr = 0.75
high_correlation_pairs = []
for i in range(len(mat.columns)):
    for j in range(i + 1, len(mat.columns)):
        if abs(mat.iloc[i, j]) > cutoff_corr:
            v1 = mat.columns[i]
            v2 = mat.columns[j]
            correlation_value = mat.iloc[i, j]
            high_correlation_pairs.append((v1, v2, correlation_value))
# for pair in high_correlation_pairs:
#     print(f"Coppia: {pair[0]} - {pair[1]}, Correlazione: {pair[2]}")
#     count_first = len(data[pair[0]].unique())
#     count_second = len(data[pair[1]].unique())
#     if count_first >= count_second:
#             drop_list.append(pair[0])
#             print(f"Della Coppia: {pair[0]} - {pair[1]}, con Correlazione: {pair[2]} elimino la Feature {pair[0]}")
#     else:
#             drop_list.append(pair[1])
#             print(f"Della Coppia: {pair[0]} - {pair[1]}, con Correlazione: {pair[2]} elimino la Feature {pair[1]}")
print(0)
