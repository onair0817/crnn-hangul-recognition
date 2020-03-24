import numpy as np
import matplotlib.pyplot as plt
import os
import editdistance
import pickle
import time
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from livelossplot.keras import PlotLossesCallback

from crnn_model import CRNN
from crnn_data import InputGenerator
from crnn_utils import decode
from utils.training import Logger, ModelSnapshot

import ph_utils
from ph_gt_data import GTUtility

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

PICKLE_DIR = './pickles/'
PICKLE_NAME = 'printed_hangul_all.pkl'
EXPERIMENT = 'crnn_lstm_ph_all_v1'
CHECKPOINT_PATH = './checkpoints/202003230905_crnn_lstm_ph_all_v1/weights.040000.h5'

# Train
train_pkl = PICKLE_DIR + os.path.splitext(os.path.basename(PICKLE_NAME))[0] + '_train.pkl'
with open(train_pkl, 'rb') as f:
    gt_util_train = pickle.load(f)

# Validation
val_pkl = PICKLE_DIR + os.path.splitext(os.path.basename(PICKLE_NAME))[0] + '_val.pkl'
with open(val_pkl, 'rb') as f:
    gt_util_val = pickle.load(f)

ph_dict = ph_utils.get_ph_dict(data_path=PICKLE_DIR, file_name=PICKLE_NAME)
print(len(ph_dict))

input_width = 256
input_height = 32
batch_size = 128
input_shape = (input_width, input_height, 1)

model, model_pred = CRNN(input_shape, len(ph_dict))
max_string_len = model_pred.output_shape[1]

gen_train = InputGenerator(gt_util_train, batch_size, ph_dict, input_shape[:2],
                           grayscale=True, max_string_len=max_string_len)
gen_val = InputGenerator(gt_util_val, batch_size, ph_dict, input_shape[:2],
                         grayscale=True, max_string_len=max_string_len)

model.load_weights(CHECKPOINT_PATH)

check_dir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + EXPERIMENT
if not os.path.exists(check_dir):
    os.makedirs(check_dir)

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# optimizer = Adam(lr=0.02, epsilon=0.001, clipnorm=1.)

# dummy loss, loss is computed in lambda layer
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

model.summary()

hist = model.fit_generator(generator=gen_train.generate(),  # batch_size here?
                           steps_per_epoch=gt_util_train.num_objects // batch_size,
                           epochs=100,
                           validation_data=gen_val.generate(),  # batch_size here?
                           validation_steps=gt_util_val.num_objects // batch_size,
                           callbacks=[
                               ModelCheckpoint(check_dir + '/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
                               ModelSnapshot(check_dir, 10000),
                               Logger(check_dir),
                           ],
                           initial_epoch=0)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
#
# epochs = range(len(loss))
# plt.figure(figsize=(15, 10))
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
