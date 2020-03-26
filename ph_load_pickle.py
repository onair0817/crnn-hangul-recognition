import pickle

from ph_gt_data import GTUtility

PICKLE_DIR = './pickles/'

# Syllable
file_name1 = 'printed_hangul_syllable.pkl'

with open(PICKLE_DIR + file_name1, 'rb') as f:
    gt_util_syllable = pickle.load(f)

# Word
file_name2 = 'printed_hangul_word.pkl'

with open(PICKLE_DIR + file_name2, 'rb') as f:
    gt_util_word = pickle.load(f)

# Sentence
file_name3 = 'printed_hangul_sentence.pkl'

with open(PICKLE_DIR + file_name3, 'rb') as f:
    gt_util_sentence = pickle.load(f)

# All
file_name4 = 'printed_hangul_all.pkl'

with open(PICKLE_DIR + file_name4, 'rb') as f:
    gt_util_all = pickle.load(f)

print(gt_util_syllable)
print(gt_util_word)
print(gt_util_sentence)
print(gt_util_all)
