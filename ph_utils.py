import re
import string
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import numpy as np

from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer

from ph_gt_data import GTUtility


def create_pickle(gt_data, output_dir, fname):
    import pickle

    # Save object to pickle file
    print(' # Save to %s ...' % (output_dir + fname))
    pickle.dump(gt_data, open(output_dir + fname, 'wb'))
    print(' # Done')


def load_pickle(data_path, file_name):

    file_path = os.path.join(data_path, file_name)

    # Load pickle file
    with open(file_path, 'rb') as f:
        gt_util = pickle.load(f)

        return gt_util


def get_ph_dict(data_path, file_name):
    gt_util = load_pickle(data_path=data_path, file_name=file_name)

    text = gt_util.text
    text = list(chain(*text))
    vect = CountVectorizer(analyzer='char').fit(text)
    charset = list(vect.vocabulary_.keys())

    pattern = '[^가-힣]'  # 한글이 아닌 문자는 공백으로 바꿔준다
    charset_dict = [re.sub(pattern, "", char) for char in charset]
    ph = [x for x in charset_dict if x != '']
    ph = "".join(ph)
    # print(ph)
    # print(len(ph))
    ph = ph + string.digits + ' _'
    # print(ph)
    # print(len(ph))

    return ph


def get_image_with_box(data_path, num=10, att_type='syllable'):
    gt_util = ''

    if att_type == 'syllable':
        gt_util = load_pickle(data_path=data_path, file_name='printed_hangul_syllable.pkl')
    elif att_type == 'word':
        gt_util = load_pickle(data_path=data_path, file_name='printed_hangul_word.pkl')
    elif att_type == 'sentence':
        gt_util = load_pickle(data_path=data_path, file_name='printed_hangul_sentence.pkl')
    elif att_type == 'all':
        gt_util = load_pickle(data_path=data_path, file_name='printed_hangul_all.pkl')
    else:
        print(' @ Error - Attribute type missing!')

    idxs = range(num)

    for idx in idxs:
        img_name = gt_util.image_names[idx]
        # img_path = os.path.join(gt_util.image_path, img_name)
        img_path = gt_util.image_path + '/' + img_name
        img = cv2.imread(img_path)

        boxes = np.copy(gt_util.data[idx][:, :-1])
        texts = np.copy(gt_util.text[idx])

        bbox = boxes.reshape(-1, 2) * [img.shape[1], img.shape[0]]

        bbox2 = bbox.reshape(-1, 8)
        bbox3 = np.ndarray.tolist(bbox2)

        xlist = []
        ylist = []
        for i in range(len(bbox3)):
            xpoint = bbox3[i][0::2]
            xpoint = xpoint + [xpoint[0]]

            ypoint = bbox3[i][1::2]
            ypoint = ypoint + [ypoint[0]]

            xlist.append(xpoint)
            ylist.append(ypoint)

        plt.figure(figsize=(20, 20))
        for i in range(len(xlist)):
            plt.plot(xlist[i], ylist[i], linewidth=3, color='r', label=texts[i])
        plt.imshow(img)
        # plt.show()
        plt.axis('off')
        plt.savefig('images/box/' + img_name)


# ph_dict = get_ph_dict(data_path='/diarl_data/hangul/', file_name='printed_hangul_all.pkl')
# get_image_with_box(data_path='/diarl_data/hangul/', num=10, att_type='all')
