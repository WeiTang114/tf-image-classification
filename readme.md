# TF-Image-Classification

# Usage

## Data preparation

You need to prepare list files of image paths and labels for training/validation/testing. The format is

```
path/to/image0.png 2
path/to/image1.png 0
path/to/image2.png 0
path/to/image3.png 1
```

## Pretrained model preparation

The pretrained AlexNet model is split to 3 files because Github has a limitation of 100MB/file. To merge them, please run

```
$ ./prepare_pretrained_alexnet.sh
```

## Training

Please modify `N_CLASSES` in globals.py for your data. 

To train at the first time, run

```
$ mkdir tmp
$ python train.py --train_dir tmp --caffemodel alexnet_imagenet.npy
```

To fine-tune, run

```
# N is your checkpoint iteration
$ python train.py --train_dir tmp/ --weights tmp/model.ckpt-N --learning-rate=0.001
```

## Testing

```
# N is your checkpoint iteration
$ python test.py --weights tmp/model.ckpt-N
``````

