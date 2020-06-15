import numpy as np
import json
import os

from ssd_data import BaseGTUtility

import ph_utils


class GTUtility(BaseGTUtility):
    """
    GT Data in COCO-Text format for printed Hangul text dataset

    """

    def __init__(self, data_path, only_with_label=True):

        super().__init__()

        self.data_path = data_path
        gt_path = data_path
        self.gt_path = gt_path
        self.image_path = os.path.join(data_path, 'images')
        self.classes = ['Background', 'Text']

        self.data_info = {}
        self.id = []
        self.text = []

        img_fnames = sorted(
            ph_utils.get_filenames(data_path, extensions=ph_utils.META_EXTENSION, recursive_=True, exit_=True))
        print(" # Total file number to be processed: {:d}.".format(len(img_fnames)))

        for idx, fname in enumerate(img_fnames):
            print(" # Start processing ... <index: {} & file name: {}>".format(idx, fname))

            # Load json file
            with open(os.path.join(gt_path, fname), encoding='UTF8') as f:
                gt_data = json.load(f)

            img_width = 0
            img_height = 0
            boxes = []
            text = []
            id = ''
            image_name = ''

            # print(" # gt_data : {}".format(gt_data))
            for item in gt_data:

                if 'text' in item:
                    text.append(item['text'])
                    # print(item['text'])
                else:
                    if only_with_label:
                        continue
                    else:
                        text.append('')

                # print(' # item : {}'.format(item))
                id = item['id']
                image_name = item['image_name']
                x = 0
                y = 0
                w = item['width']
                h = item['height']
                img_width = int(w)
                img_height = int(h)
                box = np.array([x, y, x + w, y, x + w, y + h, x, y + h], dtype=np.float32)
                boxes.append(box)

            if len(boxes) == 0:
                print(" # No bounding boxes!")
                continue

            boxes = np.asarray(boxes)
            # print(boxes.shape)
            boxes[:, 0::2] /= img_width
            boxes[:, 1::2] /= img_height

            boxes = np.concatenate([boxes, np.ones([boxes.shape[0], 1])], axis=1)

            self.id.append(id)
            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)

            print(" # Info : {} {} {} {}".format(id, image_name, boxes, text))
            print(" # ID {} is added!".format(id))

        # Create data object in COCO-Text format
        self.init()


DATA_PATH = '/diarl_data/crnn/hospital_receipt/ori_4991_aug_60000/'
# DATA_PATH = 'C:/Users/admin/dev/data/'
PICKLE_DIR = './pickles/'
FILE_NAME = 'hospital_receipt_60000.pkl'

if __name__ == '__main__':
    # Create GT data in COCO-Text format
    gt_util = GTUtility(data_path=DATA_PATH, only_with_label=True)
    ph_utils.create_pickle(gt_util, PICKLE_DIR, FILE_NAME)

    # Print contents of GT data
    print(gt_util.data)
