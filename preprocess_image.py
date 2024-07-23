import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)

        image = self.interface([image])[0]
        image = np.array(image)

        return image

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DPT():
    def __init__(self, task='depth', device='cuda'):

        self.task = task
        self.device = device

        from threestudio.utils.dpt import DPTDepthModel

        if task == 'depth':
            path = 'load/omnidata/omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else: # normal
            path = 'load/omnidata/omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

        # load model
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)


    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal
def generate_box(input_mask):
    # generate bbox
    # input_mask = img[..., 3:]
    rows = np.any(input_mask, axis=1)
    cols = np.any(input_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Create the bounding box (top-left and bottom-right coordinates)
    bbox = [col_min, row_min, col_max, row_max]

    return bbox

def recenter(img, bbox, width, height):
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2
    img_center_x, img_center_y = width / 2, height / 2
    shift_x = img_center_x - bbox_center_x
    shift_y = img_center_y - bbox_center_y
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, M, (width, height))
    return img


def padding(img):


    h,w = img.shape[:2]
    if h != w:
        max_edge = max(h,w)
        x_center = (max_edge - w) // 2
        y_center = (max_edge - h) // 2


        if len(img.shape) == 3:
            c = img.shape[2]
            padding_img = np.ones((max_edge,max_edge,c))
            padding_img[y_center:y_center + h, x_center:x_center + w, :] = img

        else:
            padding_img = np.ones((max_edge,max_edge))

            padding_img[y_center:y_center+h, x_center:x_center+w] = img


        img = padding_img
    return img


def preprocess_single_image(img_path, opt):
    out_dir = os.path.dirname(img_path)
    out_rgba = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_rgba.png')
    out_depth = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_depth.png')
    out_normal = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_normal.png')
    out_caption = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_caption.txt')

    # load image
    print(f'[INFO] loading image {img_path}...')

    # check the exisiting files
    if os.path.isfile(out_rgba) and os.path.isfile(out_depth) and os.path.isfile(out_normal):
        print(f"{img_path} has already been here!")
        return
    print(img_path)
    image= cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


    if image.shape[-1] == 4:
        carved_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = carved_image[..., -1] > 0

    # predict depth
    print(f'[INFO] depth estimation...')
    dpt_depth_model = DPT(task='depth')
    depth = dpt_depth_model(image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model

    # predict normal
    print(f'[INFO] normal estimation...')
    dpt_normal_model = DPT(task='normal')
    normal = dpt_normal_model(image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model

    recenter
    if recenter:
        print(f'[INFO] recenter...')

        final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
        final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
        final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(opt.size * (1 - opt.border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (opt.size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (opt.size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    else:
        height, width, c = img.shape
    bbox = generate_box(mask)
    height, width = image.shape[:2]

    carved_image = recenter(carved_image, bbox, width, height)
    depth = recenter(depth, bbox, width, height)
    normal = recenter(normal, bbox, width, height)

    carved_image = padding(carved_image)
    depth = padding(depth)
    normal = padding(normal)

    final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
    final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
    final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)

    final_rgba[:,:] = cv2.resize(carved_image, (opt.size, opt.size), interpolation=cv2.INTER_AREA)
    final_depth[:,:] = cv2.resize(depth, (opt.size, opt.size), interpolation=cv2.INTER_AREA)
    final_normal[:,:] = cv2.resize(normal, (opt.size, opt.size), interpolation=cv2.INTER_AREA)



        # write output
    cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(out_depth, final_depth)
    cv2.imwrite(out_normal, final_normal)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--size', default=1024, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.1, type=float, help="output border ratio")
    parser.add_argument('--recenter', action='store_true', help="recenter, potentially not helpful for multiview zero123")
    parser.add_argument('--do_caption', action='store_true', help="do text captioning")

    opt = parser.parse_args()

    sam_path = os.path.join(opt.path, "sam")
    if os.path.exists(sam_path):
        opt.path = sam_path


    img_list = sorted(os.path.join(root, fname) for root, _dirs, files in os.walk(opt.path) for fname in files)
    img_list = [img for img in img_list if not img.endswith("rgba.png") and not img.endswith("depth.png") and not img.endswith("normal.png")]
    # img_list = [img for img in img_list if img.endswith(".png")]

    for img in img_list:
        preprocess_single_image(img, opt)
