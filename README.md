# ANN_SNN_QCFS
Code for paper "Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks"

Reproducable and fix random seed. 
Use shared weight for ANN and SNN, easy to load and use.
Compatiable with old version models.

To train
L is the QCFS quantization step.
```
python main_train.py --epochs=300 -dev=0 -L=4 -data=cifar10
```

To test
T controls the simluation step of SNN. If T=0, the model act as ANN and T>0 model act as SNN.
```
python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0
```

Use default setting, a cifar10 vgg16 SNN is reported to be
* T=2, Acc=90.94
* T=4, Acc=94.01
* T=8, Acc=95.01

Use default setting (need to change lr to 0.05), a cifar100 vgg16 SNN is reported to be
* T=2, Acc=64.89
* T=4, Acc=70.42
* T=8, Acc=74.63
* T=64,Acc=77.70

If there are any bugs for this new version, pls let me know.

One pretrained model at 
https://drive.google.com/drive/folders/1P-2egAraWtsQYNzp8lcJvZVEG_KLVV5Q?usp=sharing

The CIFAR100 training configuration is updated and the example models/logs are uploaded to google drive. Sorry for take that long time.
