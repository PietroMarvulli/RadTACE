import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import SimpleITK as sitk
from torchvision import transforms, models
import torchvision.models as models

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Number of GPUs:", torch.cuda.device_count())
print("Device Name:", torch.cuda.get_device_name(0))

list_of_models = models
resnet = models.resnet50(pretrained = True,progress = True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

ct_path = "D:\\dataset_TACE_NIfTI\\HCC-TACE-Seg\\HCC_002\\ABDPELVIS_(26_09_1997)\\4_Recon_2_LIVER_3_PHASE_(AP)_acquisitionNumber_2_1.nii.gz"
mask = "D:\\dataset_TACE_NIfTI\\HCC-TACE-Seg\\HCC_002\\ABDPELVIS_(26_09_1997)\\Segmentation.nrrd"
slice_index = 205
series = sitk.ReadImage(ct_path)
series = sitk.IntensityWindowing(series, -50, 350,0,255)
new_spacing = [1, 1, 1]
original_spacing = series.GetSpacing()
original_size = series.GetSize()
new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
resample = sitk.ResampleImageFilter()
resample.SetOutputSpacing(new_spacing)
resample.SetSize(new_size)
resample.SetOutputDirection(series.GetDirection())
resample.SetOutputOrigin(series.GetOrigin())
resample.SetInterpolator(sitk.sitkLinear)
series = resample.Execute(series)
image = sitk.GetArrayFromImage(series)[slice_index]
mask = sitk.ReadImage(mask)
mask = resample.Execute(mask)
mask = sitk.GetArrayFromImage(mask)[slice_index]

######## PLOT #########
nzi = np.argwhere(mask == 2)
minimum = nzi.min(axis = 0)
max = nzi.max(axis = 0)
dim = max - minimum + 1

fig, ax = plt.subplots(1,2)
ax[0].imshow(image, cmap='gray')
ax[0].imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0)
width = max[1] - minimum[1]
height = max[0] - minimum[0]
rect1 = patches.Rectangle((minimum[1], minimum[0]), width, height, linewidth=0.5, edgecolor='red', facecolor='none')
rect2 = patches.Rectangle((minimum[1]-15, minimum[0]-15), width+30, height+30, linewidth=2, edgecolor='green', facecolor='none')
ax[0].add_patch(rect1)
ax[0].add_patch(rect2)
offset = 15
crop_image = image[minimum[0]:max[0], minimum[1]:max[1]]
crop_image_ext = image[minimum[0]-offset:max[0]+offset, minimum[1]-offset:max[1]+offset]

ax[1].imshow(crop_image_ext, cmap='gray')
ax[1].set_title("Crop Image")
plt.show()

image_3ch = np.stack([crop_image_ext] * 3, axis=-1)
img = Image.fromarray(np.uint8(image_3ch))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_tensor = preprocess(img).unsqueeze(0)
image_tensor = image_tensor.to(device)

with torch.no_grad():
    features = resnet(image_tensor)

features = features.cpu().numpy()
print("Extracted features shape:", features.shape)
flattened_features = features.flatten()
print("Flattened features shape:", flattened_features.shape)