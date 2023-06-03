import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm
from keras.callbacks import Callback
import json
from keras_unet_collection import models, base, utils, losses
from PIL import Image
import time
import datetime
import random

# enabling mixed precision, only works on GPU capability 7.0
#from keras.mixed_precision import experimental as mixed_precision
# current GPU has 6.1
# tf.keras.mixed_precision.set_global_policy("mixed_float16")
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

def masks_to_single_channel(path):
    """
        receives: path
        returns: True
        converts 3 channel masks into single channel masks
    """

    for img in tqdm(os.listdir(path)):
        re_img = cv2.imread(f'{path}/{img}')
        sh = re_img.shape
        cv2.imwrite(f'{path}_one/{img}', np.reshape(re_img[:, :, 1], (sh[0], sh[1], 1)))

def hybrid_loss(y_true, y_pred):
    """
        receives: true input tensor, prdiction input tensor
        returns: hybrid loss
        calculates the hybrid loss from FTL and IOU, adapted from keras_unet_collection
    """
    # focal tversky for class imbalance, iou for crossentropy binary
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.7, gamma=4 / 3)
    loss_iou = losses.iou_seg(y_true, y_pred)

    return loss_focal + loss_iou  # +loss_ssim

class CheckpointsCallback(Callback):
    # Callback Class, adapted from https://github.com/divamgupta/image-segmentation-keras

    def __init__(self, checkpoints_path, model_name, patience=5):
        self.checkpoints_path = checkpoints_path
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.metrics = {}
        self.model_name = model_name
        self.start_lr = 1e-4
        self.stopping_epoch = 0

    def on_train_begin(self, logs=None):
        """
            starts on beginning of each epoch, allocates variables
        """
        # define log dict
        for metric in logs:
            print(f"here are the metrics {metric}")
            self.metrics[metric] = {}
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf  # 0#np.Inf  # 0

    def on_epoch_end(self, epoch, logs=None):
        """
            updates variables and Callbacks at the end of the epoch
        """
        # Old Early Stopping 6a)
        # if epoch % 10 == 0 and epoch >= 30:
        #     self.start_lr = self.start_lr / 10
        #     print('adjusted lr to ', self.start_lr)
        #     tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.start_lr)

        # save weights
        if self.checkpoints_path is not None:
            self.model.save_weights(f'{self.checkpoints_path}/checkpoints.{str(epoch)}')
            print("saved ", self.checkpoints_path + "." + str(epoch))

        # access metrics and losses
        try:
            current_train = logs.get(f'{self.model_name}_output_final_activation_iou_seg') + logs.get(
                f'{self.model_name}_output_final_activation_focal_tversky')
            current_val = logs.get(f'val_{self.model_name}_output_final_activation_iou_seg') + logs.get(
                f'val_{self.model_name}_output_final_activation_focal_tversky')
        except:
            print('didnt find metrics')
            current_train = logs.get("loss")
            current_val = logs.get("val_loss")
        print("logs:", logs)

        # compare losses with previous epoch to compare early stopping
        if np.less(current_val, self.best):  # np.greater(current_val, self.best): #np.less(current_val, self.best):
            print(f'achieved better results.. {current_val}')
            self.best = current_val
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.stopping_epoch = epoch
        else:
            self.wait += 1
            print(f'achieved worse results than {self.best} with {current_val}')
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"Restoring model weights from the end of the best epoch {self.stopping_epoch}.")
                self.model.set_weights(self.best_weights)

        # logging data into metric
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # log epoch number to metric
        self.metrics['epoch'] = self.stopped_epoch

        # predict 10 imgs on epoch end to visually compare performance
        for arr in tqdm(imgs[:10]):
            out = np.empty((1, shape_i, shape_i, 3))
            with Image.open(f'{filepath}images/{arr}') as pixio:
                pix = pixio.resize((shape_i, shape_i), Image.Resampling.NEAREST)
                out[0, ...] = np.array(pix)[..., :3]

            image = out / 255.
            out = loaded_model.predict(x=image, verbose=0)
            img_out = np.array(out[-1]).reshape((shape_i, shape_i, 1))
            img_out2 = cv2.normalize(img_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            cv2.imwrite(f"{out_path}/{name}/epoch_results/e{epoch}_{arr}", img_out2)

    def on_train_end(self, logs=None):
        """
            wegen training ends, save logs to a file
        """

        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        # saving log to file
        self.model.set_weights(self.best_weights)
        with open(f'{self.checkpoints_path}/{self.model_name}_logs.txt', 'w') as log_file:
            log_file.write(json.dumps(str(self.metrics)))

def augment_image(image):
    """
        receives: image
        returns: augmented image
        applies random augmentations to images when loading in the dataset
    """
    # update seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random augmentations.
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    image = tf.image.stateless_random_contrast(image, 0.2, 0.5, new_seed)
    image = tf.image.stateless_random_flip_left_right(image, new_seed)
    image = tf.image.stateless_random_flip_up_down(image, new_seed)

    return image

def create_dataset(base):
    """
        receives: base path
        returns: training and validation datasets
        creates datasets for training sem. segmentation with tensorflow
    """

    image_dir = base + 'images/'
    label_dir = base + 'rehashed_ones/'
    # define the paths
    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, "*.png"), shuffle=True, seed=69)
    label_files = tf.data.Dataset.list_files(os.path.join(label_dir, "*.png"), shuffle=True, seed=69)

    # combine datasets
    dataset = tf.data.Dataset.zip((image_files, label_files))
    # apply shuffling
    dataset = dataset.shuffle(buffer_size=BATCH_SIZE)

    # create map of dataset to apply preprocessing and image import
    dataset = dataset.map(
        map_func=load_and_preprocess_image_and_label,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # split dataset
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    print(f'loaded dataset {dataset} with size {dataset_size}')
    train_size = int(TRAIN_VAL_TEST_SPLIT[0] * dataset_size)
    val_size = int(TRAIN_VAL_TEST_SPLIT[1] * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)

    # batch datasets
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # prefetch datasets for better memory handling
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset

def load_and_preprocess_image_and_label(image_path, label_path):
    """
        receives: image path, mask path
        returns: augmented image, mask both as tensor objects
        reads files and applies augmentations
    """
    # load images
    image = tf.io.read_file(image_path)
    label = tf.io.read_file(label_path)
    image = tf.image.decode_png(image, channels=3)
    label = tf.image.decode_png(label, channels=1)  # channels=3

    # normalize image
    image = tf.image.per_image_standardization(image) # tf.cast(image, tf.float32) / 255.0

    # apply augmentation on image
    return augment_image(image), label  # image_pre, label_pre #image, label

"""
def create_datasetPredict(base):
    image_dir = base + 'images/'
    # Define the paths to the image and label directories
    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, "*.png"))

    # Combine the image and label file paths into a single dataset

    # Shuffle the dataset
    # dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    # Map the dataset to load and preprocess images and labels in parallel
    dataset = image_files.map(
        map_func=load_and_preprocess_image_and_label_predict,
        num_parallel_calls=NUM_PARALLEL_CALLS
    )

    # Batch the datasets
    train_dataset = dataset.batch(BATCH_SIZE)

    # Prefetch the datasets
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset

def load_and_preprocess_image_and_label_predict(image_path):
    # Load the image and label
    image = tf.io.read_file(image_path)

    # Decode the image and label
    image = tf.image.decode_png(image, channels=3)
    # Resize the image and label
    #image = tf.image.resize(image, (shape_i, shape_i))  # )  # IMAGE_SIZE) #RESHAPE_SIZE
    #label = tf.image.resize(label, (shape_i, shape_i))  # IMAGE_SIZE)  # IMAGE_SIZE)

    # Normalize the image
    image = tf.image.per_image_standardization(image)
    # image_pre = tf.keras.applications.resnet50.preprocess_input(image)
    # label_pre = tf.keras.applications.resnet50.preprocess_input(label)

    return image
"""

"""from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K"""

# train model with booleans
train = True
predict = True
save_model = True
load_model = False
train_on = 'GPU' #'GPU'  # 'CPU'

physical_devices = tf.config.list_physical_devices(f'/device:{train_on}:0')
print(physical_devices)

# using models, losses and metrics from https://github.com/yingkaisha/keras-unet-collection/blob/main/examples/user_guide_models.ipynb
with tf.device(f'/{train_on}:0'):
    start_script = time.time()
    start_time = datetime.datetime.now()

    # Define the data directory and file paths
    out_path = 'D:/SHollendonner/segmentation_results/'
    #filepath = 'D:/SHollendonner/multispectral/channels_257/tiled512/rotated/' #/small/rotated/ #small_test_sample/'  # small_sample/' #small_test_sample/
    #pred_file_path = 'D:/SHollendonner/multispectral/channels_257/tiled512/'
    pred_file_path = 'D:/SHollendonner/tiled512/small_test_sample/'
    filepath = 'D:/SHollendonner/tiled512/small_test_sample/' #small_sample/' # small/
    name = 'U-Net_DenseNet201'

    checkpoints_path = f'{out_path}/{name}/checkpoints'
    load_name = '1305_512_unet_densenet201_MS_150epochs_small'

    # create prediction sample for each epoch
    imgs = os.listdir(f'{filepath}images/')
    random.seed(86)
    random.shuffle(imgs)

    # creating folder structure
    if not os.path.exists(f'{out_path}/{name}'):
        print('creating model folder')
        os.mkdir(f'{out_path}/{name}')
    if not os.path.exists(f'{out_path}/{name}/model'):
        print('creating model save folder')
        os.mkdir(f'{out_path}/{name}/model')
    if not os.path.exists(f'{out_path}/{name}/results'):
        print('creating results folder')
        os.mkdir(f'{out_path}/{name}/results')
    if not os.path.exists(f'{out_path}/{name}/checkpoints'):
        print('creating checkpoints folder')
        os.mkdir(f'{out_path}/{name}/checkpoints')
    if not os.path.exists(f'{out_path}/{name}/weights'):
        print('creating weights folder')
        os.mkdir(f'{out_path}/{name}/weights')
    if not os.path.exists(f'{out_path}/{name}/epoch_results'):
        print('creating epoch_results folder')
        os.mkdir(f'{out_path}/{name}/epoch_results')

    # defie hyperparamters
    IMAGE_SIZE = [512, 512]
    RESHAPE_SIZE = [256, 256]
    shape_i = 512 #256 #256  # 512
    n_labels = 1
    epochs = 200
    lr_start = 1e-4

    BATCH_SIZE = 2 #2 #10 #10 #10  # 2 #8
    NUM_PARALLEL_CALLS = 20
    TRAIN_VAL_TEST_SPLIT = [0.8, 0.2] # if a test split has been done on file before, apply this: [0.8889, 0.1111]
    PATIENCE = 7
    TF_FORCE_GPU_ALLOW_GROWTH = True

    # static seed in script
    rng = tf.random.Generator.from_seed(123, alg='philox')
    seed = rng.make_seeds(2)[0]

    if train_on == 'GPU':
        # GPU enhancment
        TF_GPU_ALLOCATOR = 'cuda_malloc_async'
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    # create the dataset
    if train:
        train_dataset, val_dataset = create_dataset(filepath)

    # model params
    type_m = 'unet_2d'
    activation = 'ReLU'
    filter_num_down = [32, 64, 128, 256, 512]
    filter_num_skip = [32, 32, 32, 32]
    filter_num_aggregate = 160
    stack_num_down = 2
    stack_num_up = 2
    filter_num = [64, 128, 256, 512, 1024]
    activation = 'ReLU'
    output_activation = 'Sigmoid'
    batch_norm = True
    pool = True
    unpool = True
    backbone = 'DenseNet201' #'EfficientNetB1' #'DenseNet121' # 'ResNet50V2' "DenseNet201",
    weights = 'imagenet'
    freeze_backbone = True
    freeze_batch_norm = True
    # compiling
    loss = hybrid_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_start)
    metrics = [hybrid_loss, losses.iou_seg]
    model_parameters = {'type_m' : type_m, 'shape_i':shape_i,'activation' : activation, 'filter_num_down' : filter_num_down,'filter_num_skip' : filter_num_skip,'filter_num_aggregate' : filter_num_aggregate,'output_activation': output_activation,'stack_num_down' : stack_num_down,'stack_num_up' : stack_num_up,'filter_num' : filter_num,'activation' : activation,'batch_norm' : batch_norm,'pool' : pool,'unpool' : unpool,'backbone' : backbone,'weights' : weights,'freeze_backbone' : freeze_backbone,'freeze_batch_norm' : freeze_batch_norm,'loss' : 'hybrid_loss','optimizer' : 'tf.keras.optimizers.Adam(learning_rate:lr_start)', 'metrics' : 'hybrid_loss'}
    with open(f'{out_path}/{name}/model_parameters.txt', 'w') as log_file:
        log_file.write(json.dumps(model_parameters))

    # unet basic
    loaded_model = models.unet_2d((shape_i, shape_i, 3), filter_num=filter_num,
                               n_labels=n_labels,
                               stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                               activation=activation,
                               output_activation=output_activation,
                               batch_norm=batch_norm, pool=pool, unpool=unpool,
                               weights=weights, backbone=backbone, # 'ResNet50V2'
                               freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm,
                               name=name)
    # unet_3plus
    """loaded_model = models.unet_3plus_2d((shape_i, shape_i, 3), n_labels=1, filter_num_down=[64, 128, 256, 512],
                           filter_num_skip='auto', filter_num_aggregate='auto',
                           stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid', weights=weights,
                           batch_norm=batch_norm, pool=pool, unpool=unpool, deep_supervision=True, backbone=backbone, name=name)

    loaded_model.compile(loss=[hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss],
                      loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      metrics=[losses.iou_seg, losses.focal_tversky])"""
    # r2unet
    """unet3plus = models.r2_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512],
                                  n_labels=1,
                                  stack_num_down=2, stack_num_up=2,
                                  recur_num=2,
                                  activation='ReLU',
                                  output_activation='Sigmoid',

                                  batch_norm=True, pool=True, unpool=True,
                                  name=name)  # , backbone='ResNet152V2'"""
    # basic attention unet
    """loaded_model = models.att_unet_2d((shape_i, shape_i, 3), filter_num=[64, 128, 256, 512, 1024], n_labels=1,
                         stack_num_down=2, stack_num_up=2,
                         activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid',
                         batch_norm=batch_norm, pool=pool, unpool=unpool, name=name, backbone=backbone, weights='imagenet',
                         freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm)"""
    # resunet
    """unet3plus = models.resunet_a_2d((256, 256, 3), [32, 64, 128, 256, 512, 1024],
                        dilation_num=[1, 3, 15, 31],
                        n_labels=1, aspp_num_down=256, aspp_num_up=128,
                        activation='ReLU', output_activation='Sigmoid',
                        batch_norm=True, pool=False, unpool='nearest', name=name)"""

    # load existing model
    if load_model:
        print("loading model")
        loading_model = utils.dummy_loader(f'{out_path}/{load_name}/model/')
        loaded_model.set_weights(loading_model)
        lr_middle = 1e-6
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_middle)

    # compile model
    loaded_model.compile(loss=loss, # tf.keras.losses.BinaryCrossentropy()
                      optimizer=optimizer,
                      metrics=metrics)

    # add callbacks
    callbacks = [CheckpointsCallback(checkpoints_path=checkpoints_path, model_name=name, patience=PATIENCE),
                 tf.keras.callbacks.ReduceLROnPlateau(patience=PATIENCE-2, monitor='val_loss', factor=0.4, verbose=1)]

    # output model summary
    loaded_model.summary()

    # train model
    if train:
        unet3plus_history = loaded_model.fit(x=train_dataset, validation_data=val_dataset, batch_size=BATCH_SIZE, epochs=epochs, verbose=1, callbacks=callbacks) # , callbacks=callbacks

    # save model
    if save_model:
        print(f'saving model to {out_path}/{name}/model/')
        loaded_model.save(f'{out_path}/{name}/model/', save_traces=True)

        # save model weights
        loaded_model.save_weights(f'{out_path}/{name}/weights/')
        print(f'saved weights to {out_path}/{name}/weights/')

    # predict segmentation output
    if predict:
        print(f'predicting output to {out_path}/{name}/results/')
        print(f'{filepath}images/')

        for arr in tqdm(os.listdir(f'{pred_file_path}images/')):
            out = np.empty((1, shape_i, shape_i, 3))
            with Image.open(f'{pred_file_path}images/{arr}') as pixio:
                pix = pixio.resize((shape_i, shape_i), Image.Resampling.NEAREST)
                out[0, ...] = np.array(pix)[..., :3]

            image = out / 255.
            out = loaded_model.predict(x=image, verbose=0)

            img_out = np.array(out[-1]).reshape((shape_i, shape_i, 1))
            img_out2 = cv2.normalize(img_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            cv2.imwrite(f"{out_path}/{name}/results/{arr}", img_out2)

    end_script = time.time()
    diff = end_script - start_script
    print(f'script took {diff}s from {start_time} to {datetime.datetime.now()}')