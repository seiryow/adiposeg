# Adipocyte Segmentation Tool using Keras

## How to Use

### Virtual env

```
workon adiposeg # or bio

```

### Prepare dataset. Your dataset should be organized like this

```
dataset_dir
	train
		label
			group1
				img1.png
				img2.png
			group2
				image1.png
				image2.png
			synthesized
				syn1.png
				syn2.png
		raw
			group1
				img1.png
				img2.png
			group2
				image1.png
				image2.png
			synthesized
				syn1.png
				syn2.png
	test
		label
			img1.png
			img2.png
			img3.png
		raw
			img1.png
			img2.png
			img3.png
	val
		label
			...
		raw
			...
```
Half of the images in `test` will be used for validation.

### Split validation set from train set

convert train.tif single%d.png

In case you have not had a validation set and want to take a part of the train set
as the validation set, you can use the following script.

```
python create_val.py data/dataset_dir #default value is 20
python create_val.py data/dataset_dir --val_ratio 10 # Take 10% of the train set
```
* raw and label shoud be identical

A new dataset named `dataset_dir_date_time` will be created.

### Preprocess

```
python preprocess.py data/dataset_dir_%date%_%time%
```

Many `.npy` files will be created and saved at `dataset_dir`

### Train

```
python train.py data/dataset_dir_%date%_%time%
```

* Final model will be saved at `dataset_dir/weights/<date>/<time>/unet.hdf5`.
* While training is still on-going, the best model upto that moment
is saved at `dataset_dir/weights/unet.hdf5`. Please be advised that you may
not have enough GPU memory for training and predicting at the same time.

### Predict

```
python predict.py data/dataset_dir_%date%_%time%
```

* The model saved at `dataset_dir/weights/unet.hdf5` will be used for prediction.
* Output (prediction results) will be saved at `dataset_dir/output/<date>/<time>/`

### Replicate a folder

```
./replicate_img.sh [source_dir] [output_dir] [number of times]
```
