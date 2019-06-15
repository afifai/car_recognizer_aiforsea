# Car Recognizer

This repository is to do car classification by fine-tuning VGG net with Cars Dataset from Stanford.
This repository is to answer Computer Vision challenge on [AI for SEA](https://www.aiforsea.com) 

We using pretrained VGG 16 and using transfer learning for predicting make and model of cars and using simple method for recognizing car color. We only defined 9 colors, there are : White, Black, Pink, Teal, Grey, Yellow, Blue, Red and Green 

## Prerequisites

First, clone this repository, then We need to download Stanford Cars Dataset, you can download it on [the official websites](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) or you can directly download it from [here](http://imagenet.stanford.edu/internal/car196/car_ims.tgz)

After we download it, extract it and put in `data/` folder, so we have following folder on `data`

```
- data
| -- car_ims
| -- checkpoint
| -- lists
| -- output
| -- rec

```

Ater that, download pretrained VGG16 model, you can download it [here](https://drive.google.com/drive/folders/1bvILxSG1rpmMSsDX0ErZyRjOT6itaZz-?usp=sharing). There are 2 files and you must download all of it. Put those files on `vgg16/` folder.

### Installation

We must install OpenCV and MxNet by building MxNet from scratch (because we need built-in mxnet tools (im2rec) in this repository). We can follow tutorial for install opencv and mxnet on this [link](https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/)

After we have an environment for running mxnet & opencv, install all prerequisites library using pip

```
pip install -r requirements
```

### Preparation for Training

If you don't want to train it from scratch, you can skip to **Inferencing model** below.

### Building Dataset

Open `car_config.py` on `config` and modify `BASE_PATH` to an absolute path to `data/`

then, run this command

```
python build_dataset.py
```

After the command finishes running, we should verify that we have `train.lst`, `test.lst`, and `val.lst` on `data/lists/`

### Create the Record Database

We need to generate mxnet record files. Remember, you must build mxnet from the source for using this command. I assume that the `mxnet` folder is on the home folder

Run this command for creating train.rec :

```
~/mxnet/bin/im2rec data/lists/train.lst ""  data/rec/train.rec resize=256 encoding='.jpg' quality=100
```

do same things for test and val set

```
~/mxnet/bin/im2rec data/lists/val.lst ""  data/rec/val.rec resize=256 encoding='.jpg' quality=100
```

```
~/mxnet/bin/im2rec data/lists/test.lst ""  data/rec/test.rec resize=256 encoding='.jpg' quality=100
```

As a sanity check, list the contents of the directory where you have stored the record databases:

```
ls -l data/rec
```

and should return this :
```
-rw-r--r-- 1 afif afif 182483508 Jun 13 23:13 test.rec
-rw-r--r-- 1 afif afif 851505800 Jun 13 23:13 train.rec
-rw-r--r-- 1 afif afif 183195852 Jun 13 23:13 val.rec
```

### Training

After we create record database for each set, we can perform training using this command

```
python fine_tune_cars.py --vgg vgg16/vgg16 --checkpoints data/checkpoints --prefix vggnet
``` 

### Training Result

This is the plot for accuracy and loss that i've got from training process

![Accuracy Plot](https://github.com/afifai/car_recognizer_aiforsea/raw/master/img/accuracy.png "Accuracy Plot")

![Accuracy Plot](https://github.com/afifai/car_recognizer_aiforsea/raw/master/img/loss.png "Loss Plot")


## Inferencing model

If you trained the model, you will have checkpoints file on `data/checkpoints` that contains parameter on every epoch and model architecture, but if you want to only inferencing our pretrained model, you must download it [here](https://drive.google.com/open?id=10_OLCLWZMHnVgHCRYJ3Sqcomc_s9DMp8) and put it on `data/checkpoints` folder

If you have `*.lst` file (see **building dataset**) you can run `viz_classification_record.py` using :

```
python vis_classification_record.py --checkpoints data/checkpoints --prefix vggnet --epoch 65 --dataset
```

But, if you only want to see prediction result using file, you can put that images file on `img_test/` folder and run this command

```
python vis_classification_file.py --checkpoints data/checkpoints --prefix vggnet --epoch 65 --dataset test_img/

```


you will get this result 

```
[INFO] actual=Mazda:Tribute
	[INFO] predicted=Mazda:Tribute, probability=42.98%
	[INFO] predicted=BMW:X5, probability=17.02%
	[INFO] predicted=Dodge:Durango, probability=6.26%
	[INFO] predicted=BMW:X3, probability=2.78%
	[INFO] predicted=Jeep:Gran-Cherokee, probability=2.70%
[INFO] predicted color=Gray

```
this is the sample results of our model :

![sample 1](https://github.com/afifai/car_recognizer_aiforsea/raw/master/res_img/res1.jpg "Sample 1")
![sample 2](https://github.com/afifai/car_recognizer_aiforsea/raw/master/res_img/res2.jpg "Sample 2")
![sample 3](https://github.com/afifai/car_recognizer_aiforsea/raw/master/res_img/res3.jpg "Sample 3")
![sample 4](https://github.com/afifai/car_recognizer_aiforsea/raw/master/res_img/res4.jpg "Sample 4")
![sample 5](https://github.com/afifai/car_recognizer_aiforsea/raw/master/res_img/res5.jpg "Sample 5")

### End Notes

We know that our color recognition is not good at all, because our methods is gather all colors from a whole image and compute the color majority, so, if the car is so small or background noise is too big, the result is wrong.

Next, we need to combine it with bounding box / background substraction for this problem
## Authors

* **Afif A. Iskandar** - Content Creator @ [NgodingPython](http://ngodingpython.com)



