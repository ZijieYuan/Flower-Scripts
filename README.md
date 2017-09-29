# Flower-Scripts
GoogLeNet Inception V3

flowers_t-sne_zj.py
  The script for creating embedding visualization 

flowers_zj.py
  The script for converting dataset (".jpg" images) to a code-readable format

inception_v3_train_val.sh
  The script for doing validation periodically during training phase

input_data_zj.py
  The script for module import

preprocessing_factory.py
  The script for saving all preprocessing functions

pre_sprite.sh
  The script for soft copy all desired images to a destination path

sec_preprocessing.py
  The script for data augmentation (8-direction rotations)

spriter.py
  The script for creating sprite image and a label file for embedding visualization

train_image_classifier_final_version.py
  The script for model training using GoogLeNet Inception V3 model

-----------------

Example:

Step1: 
	./inception_v3_train_val.sh 
	>/path/to/model/root
Step2:
	./pre_sprite.sh
	>/path/to/model/root
Step3:
	python spriter.py
	--image_dir=/path/to/dataset
	--log_dir=/path/to/log/file
Step4:
	python flowers_t-sne_zj.py
	--data_dir=/path/to/model/root
	--log_dir=/path/to/log/file
