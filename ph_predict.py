import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
import editdistance
import pickle

from crnn_model import CRNN
from crnn_data import InputGenerator
from crnn_utils import decode

import tensorflow as tf

import ph_utils
from ph_gt_data import GTUtility

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


PICKLE_DIR = './pickles/'

# AI-HUB
# PICKLE_NAME = 'printed_hangul_all.pkl'
# CHECKPOINT_PATH = './checkpoints/202004011502_crnn_lstm_ph_all_v1/weights.110000.h5'
# PLOT_NAME = 'crnn_lstm_ph_all_v1'

# AIG-IDR
# PICKLE_NAME = 'hospital_receipt_2000.pkl'
# CHECKPOINT_PATH = './checkpoints/202006031202_crnn_lstm_hr_v1/weights.004000.h5'
# PICKLE_NAME = 'hospital_receipt_5000.pkl'
# CHECKPOINT_PATH = './checkpoints/202006081103_crnn_lstm_hr_v2.2/weights.002000.h5'
PICKLE_NAME = 'hospital_receipt_60000.pkl'
CHECKPOINT_PATH = './checkpoints/202006121139_crnn_lstm_hr_v3.0/weights.014000.h5'
PLOT_NAME = 'crnn_lstm_hr_v3.0'

# Validation
# val_pkl = PICKLE_DIR + os.path.splitext(os.path.basename(PICKLE_NAME))[0] + '_val.pkl'
val_pkl = PICKLE_DIR + 'hospital_receipt_5000_val.pkl'
with open(val_pkl, 'rb') as f:
    gt_util_val = pickle.load(f)

ph_dict = ph_utils.get_ph_dict(data_path=PICKLE_DIR, file_name=PICKLE_NAME)
# print(len(ph_dict))

input_width = 256
input_height = 32
batch_size = 128
input_shape = (input_width, input_height, 1)

model, model_pred = CRNN(input_shape, len(ph_dict))

max_string_len = model_pred.output_shape[1]

gen_val = InputGenerator(gt_util_val, batch_size, ph_dict, input_shape[:2],
                         grayscale=True, max_string_len=max_string_len)

model.load_weights(CHECKPOINT_PATH)

g = gen_val.generate()
d = next(g)

res = model_pred.predict(d[0]['image_input'])

mean_ed = 0
mean_ed_norm = 0

font = {'family': 'NanumGothic',
        'color': 'black',
        'weight': 'normal',
        'size': 10,
        }

ph_utils.folder_exists('./predict_results', create_=True)

stt = time.time()

# for i in range(len(res)):
for i in range(len(res)):
    # best path, real ocr applications use beam search with dictionary and language model
    chars = [ph_dict[c] for c in np.argmax(res[i], axis=1)]
    gt_str = d[0]['source_str'][i]
    res_str = decode(chars)

    ed = editdistance.eval(gt_str, res_str)
    # ed = levenshtein(gt_str, res_str)
    ed_norm = ed / len(gt_str)
    mean_ed += ed
    mean_ed_norm += ed_norm

    # display image
    img = d[0]['image_input'][i][:, :, 0].T
    plt.figure(figsize=[10, 1.03])
    plt.imshow(img, cmap='gray', interpolation=None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.text(0, 45, '%s' % (''.join(chars)), fontdict=font)
    plt.text(0, 60, 'GT: %-24s RT: %-24s %0.2f' % (gt_str, res_str, ed_norm), fontdict=font)

    # file_name = 'plots/%s_recognition_%03d.pgf' % (plot_name, i)
    file_name = './predict_results/%s_recognition_%03d.png' % (PLOT_NAME, i)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    print(file_name)

    plt.show()

    print('%-20s %-20s %s %0.2f' % (gt_str, res_str, ''.join(chars), ed_norm))

mean_ed /= len(res)
mean_ed_norm /= len(res)

print()
print(' # Mean ED (edit-distance)             %0.3f' % mean_ed)
print(' # Mean normalized ED (edit-distance)  %0.3f' % mean_ed_norm)

print(' # Processing time : {:.2f}'.format(time.time() - stt))
