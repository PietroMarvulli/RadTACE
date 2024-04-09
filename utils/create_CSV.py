import os
import pandas as pd
main_dir = r'D:\dataset_TACE_NIfTI\HCC-TACE-SEG'

# Initialize lists to store paths
image_paths = []
mask_paths = []

# Traverse through patient folders
for patient_folder in os.listdir(main_dir):
    patient_dir = os.path.join(main_dir, patient_folder)
    if os.path.isdir(patient_dir):
        # Traverse through acquisition folders
        for acquisition_folder in os.listdir(patient_dir):
            acquisition_dir = os.path.join(patient_dir, acquisition_folder)
            if os.path.isdir(acquisition_dir):
                # Check if Segmentation.nrrd exists
                segmentation_path = os.path.join(acquisition_dir, 'Segmentation.nrrd')
                if os.path.exists(segmentation_path):
                    files = os.listdir(acquisition_dir)
                    if len(files) >= 2:
                    # Append image paths in the acquisition folder
                        file = files[-2]  # Penultimate file
                        if file.endswith('.nii.gz'):
                            image_paths.append(os.path.join(acquisition_dir, file))
                            mask_paths.append(segmentation_path)

df = pd.DataFrame({'Image': image_paths, 'Mask': mask_paths})
df.to_csv(r'D:\dataset_TACE_NIfTI\RadiomicsFeatures\image-mask_SINGLE.csv', index=False)
