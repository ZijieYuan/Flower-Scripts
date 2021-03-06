# Periodically run validation samples as its training

# user input a root path to save everything
echo "Please input a root directory for model saving:"
read path

# Where the pre-trained InceptionV3 checkpoint is saved to.
export PRETRAINED_CHECKPOINT_DIR=${path}/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
export TRAIN_DIR=${path}/flowers-models/inception_v3

export EVAL_DIR=${path}/flowers-models/inception_v3_eval

# Where the dataset is saved to.
export DATASET_DIR=${path}

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

num=100
for i in $(seq 1 16)
do
	# Fine-tune only the new layers for 1000 steps.
	python train_image_classifier_final_version.py \
	  --train_dir=${TRAIN_DIR} \
	  --dataset_name=flowers \
	  --dropout_keep_prob=0.6 \
	  --depth_multiplier=1.0 \
	  --dataset_split_name=train \
	  --dataset_dir=${DATASET_DIR} \
	  --model_name=inception_v3 \
	  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
	  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
	  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
	  --max_number_of_steps=${num} \
	  --batch_size=32 \
	  --learning_rate=0.01 \
	  --learning_rate_decay_type=fixed \
	  --save_interval_secs=60 \
	  --save_summaries_secs=60 \
	  --log_every_n_steps=100 \
	  --optimizer=rmsprop \
	  --weight_decay=0.00004

	let "num += 100"

	# Run evaluation.
	python eval_image_classifier.py \
	  --checkpoint_path=${TRAIN_DIR} \
	  --eval_dir=${EVAL_DIR} \
	  --dataset_name=flowers \
	  --dataset_split_name=validation \
	  --dataset_dir=${DATASET_DIR} \
	  --model_name=inception_v3

done
