import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.Network import DeepEMD
import tqdm
from Models.dataloader.samplers import CategoriesSampler

DATA_DIR='/home/Improved_FSIC_Method/datasets'

MODEL_DIR='/home/Improved_FSIC_Method/pretrained_model/tieredimagenet/resnet12/max_acc.pth'

parser = argparse.ArgumentParser()

#about dataset
parser.add_argument('-dataset', type=str, default='tieredimagenet', choices=['miniimagenet', 'tieredimagenet'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')

#about task
parser.add_argument('-batchsize', type=int, default=512,help='batch size of generate box')
parser.add_argument('-bs', type=int, default=1,help='batch size of tasks')
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=1,help='number of query image per class')
parser.add_argument('-test_episode', type=int, default=5000, help='number of testing episodes after training')

# about model
parser.add_argument('-model_dir', type=str, default=MODEL_DIR)
parser.add_argument('-metric', type=str, default='ADM', choices=['Cosine', 'L2', 'Dot', 'ADM'])
parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
parser.add_argument('-deepemd', type=str, default='ic', choices=['fcn', 'ic'])
parser.add_argument('-num_patch',type=int,default=16)

# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')

# OTHERS
parser.add_argument('-gpu', default='3')
parser.add_argument('-seed', type=int, default=9)

# emd_cc setting
parser.add_argument('-outer_num', type=int, default=3)
parser.add_argument('-all_in_num', type=int, default=1)
parser.add_argument('-alpha', type=float, default=1.4)

# ADM setting
parser.add_argument('-cosine_rate', type=float, default=0.8)

args = parser.parse_args()
pprint(vars(args))
set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset=set_up_datasets(args)


# model
model = DeepEMD(args)
model = load_model(model, args.model_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# test dataset
test_set = Dataset('test', args)

# Box generate
def update_box(cc_loader, model, t=0.5):
    # print('==> Start updating boxes...')
    model.eval()
    boxes = []
    for _, batch in enumerate(cc_loader):
        data, _ = [_.cuda() for _ in batch]
        with torch.no_grad():
            feat_map = model(data)  # (N, C, H, W)
        N, Cf, Hf, Wf = feat_map.shape
        eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
        eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
        eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
        eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
        eval_train_map = F.interpolate(eval_train_map, size=data.shape[-2:], mode='bilinear')  # (N, 1, Hi, Wi)

        Hi, Wi = data.shape[-2:]

        for hmap in eval_train_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)

            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    all_boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    return all_boxes

# if args.use_cc:
test_cc_loader = DataLoader(dataset=test_set, batch_size=args.batchsize, shuffle=False, num_workers=64, pin_memory=True, drop_last=False)

print('==> Start generating test boxes...')
test_set.generate_box = True

len_ds = len(test_set)
all_boxes_outer = update_box(test_cc_loader, model.module.encoder, 0.5)
assert len(all_boxes_outer) == len_ds
test_set.boxes_inner = all_boxes_outer.cpu()

test_set.generate_box = False
print('Test box generated!!!')
################################################

sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, num_workers=64, pin_memory=True)
tqdm_gen = tqdm.tqdm(loader)

# label of query images
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)

with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]  # shot: 5,3,84,84  query:5,3,84,84
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc
        m, pm = compute_confidence_interval(test_acc_record[:i])
        tqdm_gen.set_description('batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))

    m, pm = compute_confidence_interval(test_acc_record)
    result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
    result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    print(result_list[0])
    print(result_list[1])
