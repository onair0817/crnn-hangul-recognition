import re
import string
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import sys
import glob
import json
import numpy as np

from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer

RED     = (255,   0,   0)
GREEN   = (  0, 255,   0)
BLUE    = (  0,   0, 255)
CYAN    = (  0, 255, 255)
MAGENTA = (255,   0, 255)
YELLOW  = (255, 255,   0)
WHITE   = (255, 255, 255)
BLACK   = (  0,   0,   0)

IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'tiff']
VIDEO_EXTENSIONS = ['mp4', 'avi', 'mkv']
AUDIO_EXTENSIONS = ['mp3']
META_EXTENSION = ['json']
IMG_EXTENSIONS = IMAGE_EXTENSIONS
CSV_EXTENSIONS = ['csv']

COLOR_ARRAY_RGBCMY = [RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW]
COLORS = COLOR_ARRAY_RGBCMY

DEV_NULL = open(os.devnull, 'w')


def folder_exists(in_dir, exit_=False, create_=False, print_=False):
    """
    Check if a directory exists or not. If not, create it according to input argument.
    :param in_dir:
    :param exit_:
    :param create_:
    :param print_:
    :return:
    """
    if not in_dir:
        return

    if os.path.isdir(in_dir):
        if print_:
            print(" # Info: directory, {}, already existed.".format(in_dir))
        return True
    else:
        if create_:
            try:
                print(in_dir)
                os.makedirs(in_dir)
            except:
                print(" @ Error: make_dirs in check_directory_existence routine...\n")
                sys.exit()
        else:
            if print_:
                print("\n @ Warning: directory not found, {}.\n".format(in_dir))
            if exit_:
                sys.exit()
        return False


def file_exists(filename, print_=False, exit_=False):
    """
    Check if a file exists or not.
    :param filename:
    :param print_:
    :param exit_:
    :return True/False:
    """
    if not os.path.isfile(filename):
        if print_ or exit_:
            print("\n @ Warning: file not found, {}.\n".format(filename))
        if exit_:
            sys.exit()
        return False
    else:
        return True


def get_filenames(dir_path, prefixes=('',), extensions=('',), recursive_=False, exit_=False):
    """
    Find all the files rting with prefixes or ending with extensions in the directory path.
    ${dir_path} argument can accept file.
    :param dir_path:
    :param prefixes:
    :param extensions:
    :param recursive_:
    :param exit_:
    :return:
    """
    if os.path.isfile(dir_path):
        return [dir_path]

    if not os.path.isdir(dir_path):
        return []

    dir_name = os.path.dirname(dir_path)

    filenames = glob.glob(dir_name + '**/**', recursive=recursive_)
    for i in range(len(filenames)-1, -1, -1):
        basename = os.path.basename(filenames[i])
        if not (os.path.isfile(filenames[i]) and
                basename.startswith(tuple(prefixes)) and
                basename.endswith(tuple(extensions))):
            del filenames[i]

    if len(filenames) == 0:
        print(" @ Error: no file detected in {}".format(dir_path))
        if exit_:
            sys.exit(1)

    return filenames


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

    # Only for receipt
    labeled_texts = re.compile(u'[^a-zA-Z\u3131-\u3163\uac00-\ud7a3]+')
    # Only hangul
    # labeled_texts = re.compile(u'[^\u3131-\u3163\uac00-\ud7a3]+')
    charset_dict = [labeled_texts.sub(u'', char) for char in charset]
    print(" # charset_dict : {}".format(charset_dict))
    ph = [x for x in charset_dict if x != '']
    ph = "".join(ph).upper()
    ph = ph + string.digits + ' _'

    print(ph)
    print(len(ph))

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


def remove_spaces_of_labels_inside_json(dir):
    img_fnames = sorted(get_filenames(dir, extensions=META_EXTENSION, recursive_=True, exit_=True))

    for idx, fname in enumerate(img_fnames):
        # Load json file
        with open(os.path.join(dir, fname), encoding='UTF8') as f:
            gt_data = json.load(f)

        print("JSON : {}".format(gt_data))

        for idx, item in enumerate(gt_data):
            if 'text' in item:
                gt_data[idx]['text'] = str(item['text']).replace(' ', '').replace(')', '').replace('(', '').replace('/', '')

        with open(os.path.join(dir, fname), mode='w', encoding='utf-8') as f:
            json.dump(gt_data, f, ensure_ascii=False)


# remove_spaces_of_labels_inside_json(dir='/diarl_data/crnn/hospital_receipt/ori_4991_aug_60000/')
# ph_dict = get_ph_dict(data_path='./pickles/', file_name='hospital_receipt_5000.pkl')
# get_image_with_box(data_path='/diarl_data/hangul/', num=10, att_type='all')
