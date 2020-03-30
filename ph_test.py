import numpy as np
import matplotlib.pyplot as plt
import os
import editdistance
import pickle

from crnn_model import CRNN
from crnn_data import InputGenerator
from crnn_utils import decode

import ph_utils
from ph_gt_data import GTUtility

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

PICKLE_DIR = './pickles/'
PICKLE_NAME = 'printed_hangul_all.pkl'
CHECKPOINT_PATH = './checkpoints/202003271843_crnn_lstm_ph_all_v1/weights.200000.h5'
BATCH_SIZE = 100

# Validation
val_pkl = PICKLE_DIR + os.path.splitext(os.path.basename(PICKLE_NAME))[0] + '_val.pkl'
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

mean_ed = 0
mean_ed_norm = 0
mean_character_recognition_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0

word_recognition_rate = 0

j = 0
while j < BATCH_SIZE:
    d = next(g)
    res = model_pred.predict(d[0]['image_input'])

    for i in range(len(res)):
        if not j < BATCH_SIZE:
            break
        j += 1

        # best path, real ocr applications use beam search with dictionary and language model
        chars = [ph_dict[c] for c in np.argmax(res[i], axis=1)]
        gt_str = d[0]['source_str'][i]
        res_str = decode(chars)

        ed = editdistance.eval(gt_str, res_str)
        # ed = levenshtein(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm

        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.:
            correct_word_count += 1

        print('%20s %20s %f' % (gt_str, res_str, ed))

mean_ed /= j
mean_ed_norm /= j
character_recognition_rate = (char_count - sum_ed) / char_count
word_recognition_rate = correct_word_count / j

print()
print('mean editdistance             %0.3f' % mean_ed)
print('mean normalized editdistance  %0.3f' % mean_ed_norm)
print('character recognition rate    %0.3f' % character_recognition_rate)
print('word recognition rate         %0.3f' % word_recognition_rate)