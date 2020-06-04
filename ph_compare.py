import numpy as np
import os
import pytesseract as tess
import cv2
import glob
import sys
import time
import json
import tensorflow as tf

from PIL import Image
from crnn_model import CRNN
from crnn_data import crop_words
from crnn_utils import decode

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
    for i in range(len(filenames) - 1, -1, -1):
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
    elif isinstance(img_file, np.ndarray):  # not isinstance(img_file, str):
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

    img = cv2.imread(fname)

    inputs = []
    boxes = []

    x = 0
    y = 0
    w = img.shape[1]
    h = img.shape[0]
    img_width = int(w)
    img_height = int(h)
    box = np.array([x, y, x + w, y, x + w, y + h, x, y + h], dtype=np.float32)
    boxes.append(box)
    boxes = np.asarray(boxes)

    boxes[:, 0::2] /= img_width
    boxes[:, 1::2] /= img_height

    boxes = np.concatenate([boxes, np.ones([boxes.shape[0], 1])], axis=1)

    boxes = np.copy(boxes[:, :-1])

    # drop boxes with vertices outside the image
    mask = np.array([not (np.any(b < 0.) or np.any(b > 1.)) for b in boxes])
    boxes = boxes[mask]

    if len(boxes) == 0: continue

    try:
        words = crop_words(img, boxes, input_height, input_width, True)
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(fname)
        continue

    mask = np.array([w.shape[1] > w.shape[0] for w in words])
    words = [words[j] for j in range(len(words)) if mask[j]]
    if len(words) == 0: continue

    idxs_words = np.arange(len(words))
    np.random.shuffle(idxs_words)
    words = [words[j] for j in idxs_words]

    inputs.extend(words)

    images = np.ones([1, input_width, input_height, 1])
    images[0] = inputs[0].transpose(1, 0, 2)

    res = model.predict(images)

    # img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (input_height, input_width))
    # img = img[np.newaxis, :, :, np.newaxis]
    # res = model.predict(img, batch_size=128)

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
