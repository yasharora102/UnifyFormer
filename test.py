import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

from natsort import natsorted
from glob import glob
from unifyformer_arch import UnifyFormer

from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image JPEG Artical Removal using UnifyFormer')

parser.add_argument('--input_dir', default='/datasets/DIV2K_val/input/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./model_unifyformer.pth', type=str, help='Path to weights')
parser.add_argument('--tile', type=int, default=400, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--self_ensemble', type=bool, default=True, help='Self Ensemble')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

model_restoration = UnifyFormer()

weights = args.weights

checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights)

model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
result_dir  = args.result_dir
os.makedirs(result_dir, exist_ok=True)

inp_dir = args.input_dir

files = natsorted(glob(os.path.join(inp_dir, '*.png'))
                + glob(os.path.join(inp_dir, '*.PNG'))
                + glob(os.path.join(inp_dir, '*.jpg'))
                + glob(os.path.join(inp_dir, '*.JPG')))

def augment(x, self_ensemble):
    if not self_ensemble:
        return [x]
    else:
        x1 = np.copy(np.rot90(x))
        x2 = np.copy(np.rot90(x, k=2))
        x3 = np.copy(np.rot90(x, k=3))
        x4 = np.copy(np.flipud(x))
        x5 = np.copy(np.flipud(np.rot90(x)))
        x6 = np.copy(np.flipud(np.rot90(x, k=2)))
        x7 = np.copy(np.flipud(np.rot90(x, k=3)))
        return [x, x1, x2, x3, x4, x5, x6, x7]

def digment(outs, self_ensemble):
    if not self_ensemble:
        return outs
    else:
        x, x1, x2, x3, x4, x5, x6, x7 = outs
        x1 = np.copy(np.rot90(x1, k=3))
        x2 = np.copy(np.rot90(x2, k=2))
        x3 = np.copy(np.rot90(x3))
        x4 = np.copy(np.flipud(x4))
        x5 = np.copy(np.rot90(np.flipud(x5), k=3))
        x6 = np.copy(np.rot90(np.flipud(x6), k=2))
        x7 = np.copy(np.rot90(np.flipud(x7)))
        return [x, x1, x2, x3, x4, x5, x6, x7]

if args.tile is not None:
    print(f"===>Testing using tile size: {args.tile}")
    print(f"===>Testing using tile overlap: {args.tile_overlap}")
else:
    print("===>Testing on the original resolution image")

print(f"===>Testing using self_ensemble: {args.self_ensemble}")

with torch.no_grad():
    for file_ in tqdm(files):
        
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        img = np.float32(load_img(file_))/255.

        imgs = augment(img, args.self_ensemble)

        outs = []
        for img in imgs:
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            if args.tile is None:
                h,w = input_.shape[2], input_.shape[3]
                H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                restored = model_restoration(input_)

                restored = restored[:,:,:h,:w]
            else:
                b, c, h, w = input_.shape
                tile = min(args.tile, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"
                tile_overlap = args.tile_overlap

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, c, h, w).type_as(input_)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]

                        out_patch = model_restoration(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                        W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                restored = E.div_(W)

            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            outs.append(restored)

        outs = digment(outs, args.self_ensemble)
        outs = np.stack(outs)
        outs = outs.mean(0)

        outs = np.clip(outs,0.,1.)
        save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(outs))
