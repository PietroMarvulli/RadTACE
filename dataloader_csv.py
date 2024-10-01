import os
import pandas as pd


data = pd.read_csv('file\\clinical_data_II.csv')
data = data[['ID','target']]
image_dir = "D:\\dataset_5f_patched\\fold_1\\train"
image_paths = []
labels = []

for root, dirs, files in os.walk(image_dir):
    for el in dirs:
        if el in data['ID'].to_string():
            label = data[data['ID']==el]['target'].values[0]
            image_paths.append(os.path.join(root,el))
            labels.append(label)
final_df = pd.DataFrame({'directory': image_paths, 'label': labels})
final_df.to_csv('dataset_train.csv', index=False)