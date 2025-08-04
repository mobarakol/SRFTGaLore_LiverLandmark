# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from utils.metrics import evaluation
from utils import prepare_dataset
from utils.dataloader import dataset, save_and_show_img
from models.model import Model

import numpy as np



def main(save_path, args):
    val_transform = T.Compose([
        T.ToTensor()
    ])

    test_dataset = dataset(root=(args.data_path+'/Test'), transform=val_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda")
    print('test sample size:', len(test_dataset))

    model = Model(1024, 1024).to(device)
    model_checkpoint = torch.load(args.model_path)
    model.load_state_dict(model_checkpoint)

    model.eval()

    validation_IOU = []
    mDice = []

    
    for X_batch, depth, y_batch, name in tqdm(test_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        depth = depth.to(device)

        output, feature = model(X_batch, depth)
        output = F.interpolate(output, size=(1024, 1024), mode='nearest').clone()
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        y_batch = torch.argmax(y_batch, dim=1)

        tmp2 = y_batch.detach().cpu().numpy()
        tmp = output.detach().cpu().numpy()
        tmp = tmp[0]
        tmp2 = tmp2[0]

        pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
        gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

        iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

        gt_mask =np.array([tmp == i for i in range(4)]).astype(bool)
        pred_mask = np.array([tmp2 == i for i in range(4)]).astype(bool)
        gt_mask =  gt_mask[0]
        pred_mask = pred_mask[0]



        validation_IOU.append(iou)
        mDice.append(dice)

        # Save image
        save_dir = save_path
        filename_base = os.path.splitext(os.path.basename(name[0]))[0]
        img_save_path = os.path.join(save_dir, f"{filename_base}_compare.png")

        save_and_show_img(
            image=np.transpose(X_batch[0].cpu().numpy(), (1, 2, 0)) * 255,
            pred_mask=output[0].detach().cpu().numpy(),
            gt_mask=y_batch[0].detach().cpu().numpy(),
            filename=name[0],
            loss=dice.item(),
            save_path=img_save_path
        )

    print("Validation IOU:", np.mean(validation_IOU))
    print("Mean Dice:", np.mean(mDice))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--data_path', type=str, default='L3D_Dataset')
    args = parser.parse_args()
    os.makedirs('test_results/', exist_ok=True)

    save_path = 'test_results/'
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)
