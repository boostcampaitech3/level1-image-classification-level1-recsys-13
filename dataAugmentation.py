import os
import cv2
import cvlib as cv
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

class dataAugmentation:
    def __init__(self, pre_df : pd.DataFrame, train_df : pd.DataFrame):
        self.pre_df = pre_df.copy()
        self.train_df = train_df.copy()

    def image_mixup(self, image_path_li):
        image_li = []
        for image_path in image_path_li:
            image = np.array(Image.open(image_path)).astype(np.int16)
            image_li.append(image)
        genereted_image = np.mean(image_li, axis=0).astype(np.uint8)

        return genereted_image

    def image_face_crop(self, image):
        face, confidence = cv.detect_face(image)
        if not face: return 0
        x, y, w, h = face[0]
        H, W, C = image.shape
        image = image[max(y - 50, 0): min(h + 50, H), max(0, x - 50): min(w + 50, W)]
        return image

    def make_face_crop_image(self, config):
        facecrop_image_dir = config.train_facecrop_image_dir
        if not os.path.isdir(facecrop_image_dir):
            os.mkdir(facecrop_image_dir)

        for line in tqdm(self.pre_df.iloc):
            facecrop_image_id_dir = facecrop_image_dir + '/' + line['path']
            if not os.path.isdir(facecrop_image_id_dir):
                os.mkdir(facecrop_image_id_dir)
            for i in self.train_df.set_index('id').loc[line['id'], :].iloc:
                facecrop_image_path = facecrop_image_id_dir + '/' + i['path'].split('/')[-1]

                row_image_path = i['path']
                row_image = np.array(Image.open(row_image_path))

                facecrop_image = self.image_face_crop(row_image)
                if type(facecrop_image) != type(0):
                    Image.fromarray(facecrop_image, 'RGB').save(facecrop_image_path)

        print('Face Crop Image End!')