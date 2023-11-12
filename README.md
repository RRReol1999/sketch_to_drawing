# Sketch_to_drawing

This project is based on [SyTr^2 : Image Style Transfer with Transformers](https://arxiv.org/abs/2105.14576).

## Results presentation 
<p align="center">
<img src="https://github.com/RRReol1999/sketch_to_drawing/56400.jpg" width="90%" height="90%">
</p>
  After normalization of the inputs and modification of the weights of the loss function, replacing the stylized images with sketches presented the result like this.<br>


## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test.py  --content_dir your_contentdir --style_dir your_styledir    --output your_outdir
```
### Training  
Style dataset is selected uncolored parts from https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair/data  <br>  
content dataset is selected uncolored parts from https://www.kaggle.com/datasets/chaosinism/anime-sketch-pairs-from-tweets?rvi=1  <br>  
```
!python ./drive/MyDrive/StyTR-2-main/train.py --style_dir ./drive/MyDrive/StyTR-2-main/input/mixsty/twitter --content_dir ./drive/MyDrive/StyTR-2-main/input/content --save_dir ./drive/MyDrive/input/test --batch_size 2
```

