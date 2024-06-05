# SFE-Net

Deep Learning-based Optical Aberration Estimation Enables Offline Digital Adaptive Optics and Super-resolution Imaging  
By Chang Qiao, Haoyu Chen, Run Wang, Tao Jiang, and Dong Li


## Set up environment

python version 3.8  
`$ pip install requirements.txt`  
This project use wandb to log training progress, please log in with your API key get from https://wandb.ai/:  
`$ wandb login [Your API key]`  
, or you can change training log code in ./codes/trains/train_*.py

## Usage


train SFE-Net on BioSR dataset    
`python ./codes/trains/train_SFENet.py --opt ./options/trains/train_SFENet.yaml`
***
run pre-trained SFE-Net to check performance on validation BioSR data  
`python ./codes/tests/test_SFENet.py --opt ./options/tests/test_SFENet.yaml`    

run KernelGAN or pre-trained IKC/MANet to check performance on validation BioSR data  
`python ./codes/trains/train_KernelGAN.py --opt ./options/trains/train_KernelGAN.yaml`  
`python ./codes/tests/test_IKC.py --opt ./options/tests/test_IKC.yaml`  
`python ./codes/tests/test_MANet.py --opt ./options/tests/test_MANet.yaml`  

set `['test_data']['preload_data']` in `yaml` to check different performance on different polynomials ranging of zernike introduced in images.

## Demonstration

### Fig.1.Network Architecture of Spatial-frequency Encoding Network

![](figures/Fig.%201.%20Network%20Architecture%20of%20Spatial-frequency%20Encoding%20Network.png)

### Fig. 2. Network Architecture of Spatial Feature Transform-guided Deep Fourier Channel Attention Network (SFT-DFCAN)

![](figures/Fig.%202.%20Network%20Architecture%20of%20Spatial%20Feature%20Transform-guided%20Deep%20Fourier%20Channel%20Attention%20Network%20(SFT-DFCAN).png)

### Fig. 3. Schematic of the data augmentation and training process of SFE-Net

![](figures/Fig.%203.%20Schematic%20of%20the%20data%20augmentation%20and%20training%20process%20of%20SFE-Net.png)

### Fig. 4. Optical Aberration Estimation via SFE-Net

![](figures/Fig.%204.%20Optical%20Aberration%20Estimation%20via%20SFE-Net.png)

### Fig. 5. Progression of training loss and validation PSNR of network model with/without frequential branch during training process

![](figures/Fig.%205.%20Progression%20of%20training%20loss%20and%20validation%20PSNR%20of%20network%20model.png)

### Fig. 6. Blind Deconvolution with Estimated PSF

![](figures/Fig.%206.%20Blind%20Deconvolution%20with%20Estimated%20PSF.png)

### Fig. 7. Aberration-aware Image Super-resolution Reconstruction with Estimated PSF

![](figures/Fig.%207.%20Aberration-aware%20Image%20Super-resolution%20Reconstruction%20with%20Estimated%20PSF.png)

### Fig. 8. Digital Adaptive optics and Super-resolution for Live-cell Imaging

![](figures/Fig.%208.%20Digital%20Adaptive%20optics%20and%20Super-resolution%20for%20Live-cell%20Imaging.png)
