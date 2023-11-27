# Sketch_to_drawing

This project is based on [SyTr^2 : Image Style Transfer with Transformers](https://arxiv.org/abs/2105.14576).

## Results presentation 
<p align="center">
<img src="https://github.com/RRReol1999/sketch_to_drawing/blob/main/56400.jpg" width="90%" height="90%">
</p>
  After normalization of the inputs and modification of the weights of the loss function, replacing the stylized images with sketches presented the result like this.<br>
  This set of images is the result when batch size=2, numbered (1)-(6) from left to right. Where (1) and (2) are style images. (3) and (4) are content images. (5) represents the result of combining (1) and (3), and (6) represents the result of combining (2) and (4).<br>


## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Steps
If you are using google colab for project deployment, you can refer to the following steps to configure.<br> 
```
%%bash
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
```
```
%%bash
conda install --channel defaults conda python=3.6 --yes
conda update --channel defaults --all --yes
```
```
import sys
sys.path
```
```
_ = (sys.path.append("/usr/local/lib/python3.6/site-packages"))
_ = (sys.path.append("/usr/local/lib/python3.6/dist-packages"))
```
```
!pip install tensorboardX
!pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test.py  --content_dir your_contentdir --style_dir your_styledir    --output your_outdir
```
### Training
Style dataset is selected uncolored parts from https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair/data  <br>  
content dataset is selected uncolored parts from https://www.kaggle.com/datasets/chaosinism/anime-sketch-pairs-from-tweets?rvi=1  <br>  
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing)<br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
!python train.py --style_dir your_styledir --content_dir your_contentdir --save_dir your_savedir --batch_size 8
```
### To do
1.  Modify the model to enable no style image input in the test process<br>
2.  Continue to modify the loss function of style part for stylized images to expand attention to line detail<br>