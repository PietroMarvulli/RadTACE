import os
import pandas as pd

rad_features = pd.read_csv(r"file/RadFeatures_Lesion.csv")
cols_to_drop = [col for col in rad_features.columns if 'diagnostic' in col]
cols_to_drop.append("Mask")
rad_features = rad_features.drop(columns=cols_to_drop)
rad_features["Image"] = rad_features["Image"].str[35:42]
clinical_feature = pd.read_excel(r"file/HCC-TACE-Seg_clinical_data-V2.csv")
print(0)

