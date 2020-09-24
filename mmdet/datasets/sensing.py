from .custom import CustomDataset
from .builder import DATASETS
import numpy as np
import json


@DATASETS.register_module
class SensingDataset(CustomDataset):
    CLASSES = ('bridge', 'oilcan', 'port','ship','plane')

    def load_annotations(self, ann_file):
        '''
        load annotations from .json ann_file
        '''
        self.img_infos = []
        self.ann_infos = []
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        for img_data in self.data['images']:
            self.img_infos.append({
                'filename': img_data['filename'],
                'height': img_data['height'],
                'width': img_data['width'],
                'id': img_data['id']
            })
            self.ann_infos.append({
                'bboxes': np.empty((0, 4)) if len(img_data['annotation']['bboxes']) == 0 else np.array(img_data['annotation']['bboxes'], dtype=np.float32),
                'bboxes_ignore': np.empty((0, 4), dtype=np.float32),
                'labels': np.array(img_data['annotation']['labels'], dtype=np.int64),
                'labels_ignore': np.empty((0,), dtype=np.int64)
            })
            
        return self.img_infos

    def get_ann_info(self, idx):
        return self.ann_infos[idx]

    def get_cat_ids(self, idx):
        return self.ann_infos[idx]['labels'].tolist()
