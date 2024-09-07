import glob
import os
import shutil

from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tqdm
from PIL import Image


def generate_mask(image, box, save_path):
    predictor.set_image(image)
    box = [int(b) for b in box]
    box = np.array(box)
    x_c = (box[0] + box[2]) // 2
    y_c = (box[1] + box[3]) // 2
    input_point = np.array([[x_c, y_c]])
    input_label = np.array([1])

    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=False,)

    mask = mask.astype(np.uint8) * 255
    mask = np.transpose(mask, (1, 2, 0))
    img_rgba = np.concatenate([image, mask], axis=2)

    img_rgba = Image.fromarray(img_rgba, "RGBA")
    img_rgba.save(save_path)


device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)


if __name__ == '__main__':
    # root = "scene-0103"
    root = "scene-0916"
    # id_list=["3620feb00d744241a94855f3a8913a78"]


    # root = "scene-0655"
    # id_list=["94b33ce331b844dcb991a2020742cebf"]
    # id_list=["32e7ed87deb6491685b5c621c6db9b66"]
    id_list = os.listdir(root)

    for id in tqdm.tqdm(id_list):
        img_dir = os.path.join(root,id,"images")
        save_img_dir = os.path.join(root,id,"sam")
        os.makedirs(save_img_dir, exist_ok=True)

        img_filenames = os.listdir(img_dir)
        boxinfos = open(os.path.join(root, id, "track_info.txt")).readlines()

        for img_name in img_filenames:
            img_path = os.path.join(img_dir,img_name)
            img_index = int(img_name.split(".")[0])

            bbox = []
            for boxdata in boxinfos[1:]:
                boxdata = boxdata.strip().split()
                if int(boxdata[0]) == img_index:
                    bbox = [boxdata[-5], boxdata[-4], boxdata[-3], boxdata[-2]]


            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            generate_mask(image, bbox, os.path.join(save_img_dir, img_name))