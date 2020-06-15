import pickle
import numpy as np

from ssd_data import BaseGTUtility
from ph_gt_data import GTUtility


def random_split(self, split=0.8):
    gtu1 = BaseGTUtility()
    gtu1.gt_path = self.gt_path
    gtu1.image_path = self.image_path
    gtu1.classes = self.classes

    gtu2 = BaseGTUtility()
    gtu2.gt_path = self.gt_path
    gtu2.image_path = self.image_path
    gtu2.classes = self.classes

    n = int(round(split * len(self.image_names)))

    idx = np.arange(len(self.image_names))

    np.random.seed(0)

    np.random.shuffle(idx)

    train = idx[:n]
    val = idx[n:]

    gtu1.image_names = [self.image_names[t] for t in train]
    gtu2.image_names = [self.image_names[v] for v in val]
    gtu1.data = [self.data[t] for t in train]
    gtu2.data = [self.data[v] for v in val]

    if hasattr(self, 'text'):
        gtu1.text = [self.text[t] for t in train]
        gtu2.text = [self.text[v] for v in val]

    gtu1.init()
    gtu2.init()
    return gtu1, gtu2


PICKLE_DIR = './pickles/'

# AI-HUB
# PICKLE = 'printed_hangul_word.pkl'
# TRAIN = 'printed_hangul_word_train.pkl'
# VALIDATION = 'printed_hangul_word_val.pkl'

# AIG-IDR
PICKLE = 'idr_receipt_60000.pkl'
TRAIN = 'idr_receipt_60000_train.pkl'
VALIDATION = 'idr_receipt_60000_val.pkl'

with open(PICKLE_DIR + PICKLE, 'rb') as f:
    gt_util_cracker = pickle.load(f)

gt_util_train, gt_util_val = random_split(gt_util_cracker)

print(' # Train pkl file saves to %s ...' % TRAIN)
pickle.dump(gt_util_train, open(PICKLE_DIR + TRAIN, 'wb'))
print(' # Done')

print(' # Validation pkl file saves to %s ...' % VALIDATION)
pickle.dump(gt_util_val, open(PICKLE_DIR + VALIDATION, 'wb'))
print(' # Done')

print(len(gt_util_train.image_names))
print(len(gt_util_val.image_names))
