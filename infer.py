import os
import sys
import glob
import random
import timeit
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from source.dataset import TestDataset
#from model import C3D_model, R2Plus1D_model, R3D_model

from source.model.utils.vit import TimeSformer

cls_li = ['driveway_walk', 'fall_down', 'fighting', 'jay_walk', 'normal', 'putup_umbrella', 'ride_cycle', 'ride_kick', 'ride_moto']
 
# Use GPU if available else revert to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'assets')

dataset = 'kids'
num_classes = 9
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
batch_size = 4


def infer(model_path='base_weights.pth.tar', random_seed=42):

    RANDOM_SEED = random_seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    #model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
    model = TimeSformer(
        img_size=args.img_size,
        num_classes=num_classes,
        num_frames=args.num_frames,
        attention_type=args.attention_type,
        pretrained_model=args.pretrained_model,
    )
    
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    print(f"Initializing weights from: {model_path.split('/')[-1]}...")
    model.load_state_dict(checkpoint['state_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    print('Model Inference on {} dataset...'.format(dataset))

    if os.path.isdir(os.path.join(DATA_DIR, 'test_processed')):
        preprocess = False
    else:
        preprocess = True

    test_dataset = TestDataset(root_dir = "../dataset", dataset=dataset, clip_len=16, preprocess=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model.eval()
    start_time = timeit.default_timer()

    pred_li = []
    for inputs in tqdm(test_dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        pred_li.extend(preds.tolist())

    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    sample_submission = pd.read_csv('../sample_submission.csv')
    sample_submission['class'] = [cls_li[int(pred)] for pred in pred_li]
    sample_submission.to_csv('submit.csv', index=False)


if __name__ == "__main__":

    model_path = './run/run_15/models/TimeSformer-kids_epoch-8_epoch_score-0.767767.pt.pth.tar'
    infer(model_path)

