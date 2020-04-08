import numpy as np
import os
import pytesseract as tess
import cv2
import glob
import sys
import re
import time
import json
import pickle

from PIL import Image
from crnn_model import CRNN
from crnn_data import InputGenerator
from crnn_utils import decode

import ph_utils
from ph_gt_data import GTUtility

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


def split_fname(fname):
    """
    Split the filename into folder, core name, and extension.
    :param fname:
    :return:
    """
    folder = os.path.dirname(fname)
    base_fname = os.path.basename(fname)
    split = os.path.splitext(base_fname)
    core_fname = split[0]
    ext = split[1]
    return folder, core_fname, ext


def imread(img_file, color_fmt='RGB'):
    """
    Read image file.
    Support gif and pdf format.
    :param  img_file:
    :param  color_fmt: RGB, BGR, or GRAY. The default is RGB.
    :return img:
    """
    if isinstance(img_file, str):
        pass
    elif isinstance(img_file, np.ndarray):     # not isinstance(img_file, str):
        # print(" % Warning: input is NOT a string for image filename")
        # 이 경우는 img_file 이 파일 이름이 아니고 numpy array 일 경우 img_file 을 return 하는 기능이다.
        # 따라서 None 을 return 하지 말고 img_file 이 numpy array 인지를 check 하도록 수정하는 것이 좋다.
        # if 구성의 completeness를 위해 string 도 아니고 numpy array 도 아닌 경우에는 None 을 return 하도록 추가했다.
        # return None
        return img_file
    else:
        return None

    if not os.path.exists(img_file):
        print(" @ Error: image file not found {}".format(img_file))
        return None

    if not (color_fmt == 'RGB' or color_fmt == 'BGR' or color_fmt == 'GRAY'):
        color_fmt = 'RGB'

    if img_file.split('.')[-1] == 'gif':
        gif = cv2.VideoCapture(img_file)
        ret, img = gif.read()
        if not ret:
            return None
    else:
        # img = cv2.imread(img_file.encode('utf-8'))
        # img = cv2.imread(img_file)
        # img = np.array(Image.open(img_file.encode('utf-8')).convert('RGB'), np.uint8)
        img = np.array(Image.open(img_file).convert('RGB'), np.uint8)
    if img is None:
        return None

    if color_fmt.upper() == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color_fmt.upper() == 'BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return img


PICKLE_DIR = './pickles/'
PICKLE_NAME = 'printed_hangul_all.pkl'
CHECKPOINT_PATH = './checkpoints/202004011502_crnn_lstm_ph_all_v1/weights.110000.h5'
BATCH_SIZE = 1000

HANGUL_CON_VOW_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ',
                       'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ',
                       'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']

with open(os.path.join('/diarl_data/hangul/', 'printed_data_info.json'), encoding='UTF8') as f:
    gt_data = json.load(f)

data_info = gt_data['info']

# crnn references
ph_dict = ph_utils.get_ph_dict(data_path=PICKLE_DIR, file_name=PICKLE_NAME)

input_width = 256
input_height = 32
batch_size = 128
input_shape = (input_width, input_height, 1)

# model, model_pred = CRNN(input_shape, len(ph_dict))
model = CRNN((input_width, input_height, 1), len(ph_dict), prediction_only=True)

model.load_weights('./checkpoints/202004011502_crnn_lstm_ph_all_v1/weights.110000.h5')

# tesseract references
lang = 'kor'
tess_cfg = " --psm 6 --oem 1 --tessdata-dir tessdata/org"

img_fnames = sorted(get_filenames('/home/sungsoo/Downloads/WORDS/', extensions='png', recursive_=True, exit_=True))

start_time = time.time()
corr_cnt = 0

for idx, fname in enumerate(img_fnames):
    dir_name, core_name, ext = split_fname(fname)

    ans = ''
    if 'annotations' in gt_data.keys():
        for item in gt_data['annotations']:
            if item['id'] == core_name:
                ans = item['text']

    img = Image.open(fname)
    # convert image to numpy array
    data = np.asarray(img)
    print(type(data))
    # summarize shape
    print(data.shape)

    res = model.predict(np.resize(data, (128, 256, 32, 1)))

    print(type(res))

    res_str = ''

    for i in range(len(res)):
        chars = [ph_dict[c] for c in np.argmax(res[i], axis=1)]
        res_str = decode(chars)

    # res = tess.image_to_string(img, lang=lang, config=tess_cfg)
    # res = res.replace('\n', '')
    # res = re.compile(u'[^a-zA-Z\u3131-\u3163\uac00-\ud7a3]+').sub(u' ', res)
    # res = ''.join([l for l in res if l not in HANGUL_CON_VOW_LIST])
    # res = res.replace(' ', '')
    print("{} {} : {}".format(idx, ans, res_str))

    if ans == res_str:
        corr_cnt = corr_cnt + 1

    if int(idx) >= BATCH_SIZE:
        print(" # Total {} / Correct {}".format(idx, corr_cnt))
        break

print(" # Total time : {:.2f}".format(time.time() - start_time))

