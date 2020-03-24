import numpy as np
import json
import os

from ssd_data import BaseGTUtility

import ph_utils


class GTUtility(BaseGTUtility):
    """
    GT Data in COCO-Text format for printed Hangul text dataset

    """

    def __init__(self, data_path, att_type='syllable', only_with_label=True):

        super().__init__()

        self.data_path = data_path
        gt_path = data_path
        self.gt_path = gt_path
        self.image_path = ''
        self.classes = ['Background', 'Text']

        self.data_info = {}
        self.id = []
        self.text = []

        # Load json file
        with open(os.path.join(gt_path, 'printed_data_info.json'), encoding='UTF8') as f:
            gt_data = json.load(f)

        type_name = ''

        if att_type == 'syllable':
            self.image_path = os.path.join(data_path, 'printed_syllable_images')
            type_name = '글자(음절)'
        elif att_type == 'word':
            self.image_path = os.path.join(data_path, 'printed_word_images')
            type_name = '단어(어절)'
        elif att_type == 'sentence':
            self.image_path = os.path.join(data_path, 'printed_sentence_images')
            type_name = '문장'
        elif att_type == 'all':
            self.image_path = os.path.join(data_path, 'printed_images')
            type_name = '전체'
        else:
            print(' @ Error - Attribute type missing!')

        self.data_info = gt_data['info']

        # Extract a necessary value from GT data
        if 'annotations' in gt_data.keys():

            for item in gt_data['annotations']:

                img_width = 0
                img_height = 0
                boxes = []
                text = []

                if item['attributes']['type'] == type_name or type_name == '전체':
                    if 'text' in item:
                        text.append(item['text'])
                        # print(item['text'])
                    else:
                        if only_with_label:
                            continue
                        else:
                            self.text.append('')

                    for idx in range(len(gt_data['images'])):
                        if item['id'] in gt_data['images'][idx]['id']:
                            x = 0
                            y = 0
                            w = gt_data['images'][idx]['width']
                            h = gt_data['images'][idx]['height']
                            img_width = int(w)
                            img_height = int(h)
                            box = np.array([x, y, x + w, y, x + w, y + h, x, y + h], dtype=np.float32)
                        else:
                            continue

                        boxes.append(box)

                    if len(boxes) == 0:
                        print(" # No bounding boxes!")
                        continue

                    boxes = np.asarray(boxes)
                    # print(boxes.shape)
                    boxes[:, 0::2] /= img_width
                    boxes[:, 1::2] /= img_height

                    boxes = np.concatenate([boxes, np.ones([boxes.shape[0], 1])], axis=1)

                    self.id.append(item['id'])
                    self.image_names.append(item['image_id'] + '.png')
                    self.data.append(boxes)
                    self.text.append(text)

                    print(" # ID {} is added!".format(item['id']))

        # Create data object in COCO-Text format
        self.init()


ATTRIBUTE_TYPE = 'all'
DATA_PATH = '/diarl_data/hangul/'
PICKLE_DIR = './pickles/'
FILE_NAME = 'printed_hangul_all.pkl'

if __name__ == '__main__':
    # Create GT data in COCO-Text format
    gt_util = GTUtility(data_path=DATA_PATH, att_type=ATTRIBUTE_TYPE, only_with_label=True)

    ph_utils.create_pickle(gt_util, PICKLE_DIR, FILE_NAME)

    # Print contents of GT data
    print(gt_util.data)
