# Super_Pix_Adv
Offical implemention of Robust Superpixel-Guided Attentional Adversarial Attack (CVPR2020)

Setup
-----

* Install ``python`` -- This repo is tested with ``3.6``

* Install ``PyTorch version >= 1.0.0, torchvision >= 0.2.1``

* Download Inception-v3 pretrained model from ``https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth`` and move it to ``./PRETRAIN/``

Super-Pixel Generation
------------------
	python -u get_slic.py

K: number of super pixels
M: compactness factor

Adversarial Attack
------------------
run on ImageNet dataset for white-box attack succes rate and robustness toward resize(2x)
	
	python -u abim_sup.py
	
