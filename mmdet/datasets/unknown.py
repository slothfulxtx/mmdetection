from .custom import CustomDataset
from .builder import DATASETS
import numpy as np
import json


@DATASETS.register_module
class UnknownDataset(CustomDataset):
    CLASSES = ('plane', 'ship', 'vehicle')

    def load_annotations(self, ann_file):
        '''
        load annotations from .json ann_file
        '''
        self.img_infos = []
        self.img_names = []
        self.ann_infos = {}
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        for img_data in self.data['images']:
            self.img_infos.append({
                'filename': img_data['file_name'],
                'height': img_data['height'],
                'width': img_data['width'],
                'id': img_data['id']
            })
            self.ann_infos[str(img_data['id'])] = {
                'category_id': [],
                'bbox': [],
                'area': [],
            }
            self.img_names.append(img_data['file_name'])

        for ann_data in self.data['annotations']:
            if str(ann_data['image_id']) in self.ann_infos:
                self.ann_infos[str(ann_data['image_id'])]['category_id'].append(
                    ann_data['category_id'])
                self.ann_infos[str(ann_data['image_id'])]['bbox'].append(
                    ann_data['bbox'])
                self.ann_infos[str(ann_data['image_id'])]['area'].append(
                    ann_data['area'])

        return self.img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_data = self.ann_infos[str(img_id)]
        ann = {}

        bboxes = np.empty((0, 4)) if len(
            ann_data['bbox']) == 0 else np.array(ann_data['bbox'], dtype=np.float32)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        ann['bboxes'] = bboxes
        ann['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
        ann['labels'] = np.array(ann_data['category_id'], dtype=np.int64)
        ann['labels_ignore'] = np.zeros((0,), dtype=np.int64)
        # print(ann)
        return ann

    def get_cat_ids(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_data = self.ann_infos[str(img_id)]
        return ann_data['category_id']
