import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.Network import DeepEMD
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt

PRETRAIN_DIR='pretrained_model/'

DATA_DIR='/home/Improved_FSIC_Method/datasets'

parser = argparse.ArgumentParser()
#about dataset and training
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet','tieredimagenet'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')

#about training
parser.add_argument('-batchsize', type=int, default=512,help='batch size of generate box')
parser.add_argument('-bs', type=int, default=1,help='batch size of tasks')
parser.add_argument('-max_epoch', type=int, default=30)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-val_frequency',type=int,default=50)
parser.add_argument('-random_val_task', type=int, default=1, help='random samples tasks for validation at each epoch')
#about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=1,help='number of query image per class')
parser.add_argument('-val_episode', type=int, default=2000, help='number of validation episode')
parser.add_argument('-test_episode', type=int, default=5000, help='number of testing episodes after training')

# about model
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
parser.add_argument('-metric', type=str, default='ADM', choices=['Cosine', 'L2', 'Dot', 'ADM'])
parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
parser.add_argument('-deepemd', type=str, default='ic', choices=['fcn', 'ic'])
parser.add_argument('-num_patch',type=int,default=9)

# emd_cc setting
parser.add_argument('-outer_num', type=int, default=2)
parser.add_argument('-all_in_num', type=int, default=1)
parser.add_argument('-alpha', type=float, default=0.4)

# ADM setting
parser.add_argument('-cosine_rate', type=float, default=0.8)

# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')

# OTHERS
parser.add_argument('-gpu', default='0')
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-train_time', default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), type=str, metavar='PATH', help='path to cache (default: none)')

args = parser.parse_args()
pprint(vars(args))

set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset=set_up_datasets(args)

# model
args.pretrain_dir = osp.join(args.pretrain_dir, args.dataset, 'resnet12/max_acc.pth')
print(args.pretrain_dir)
model = DeepEMD(args)
model = load_model(model, args.pretrain_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

args.save_path = '%s/%s/%dquery-%dshot-%dway/'%(args.dataset,args.deepemd,args.query,args.shot,args.way)

args.save_path = osp.join('checkpoint', args.save_path, args.train_time)

ensure_path(args.save_path)

############################################
# generate box
############################################

# Dataset prepare
trainset = Dataset('train', args)
valset = Dataset('val', args)

# Upadate box of dataset
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

# Box generate
train_cc_loader = DataLoader(dataset=trainset, batch_size=args.batchsize, shuffle=False, num_workers=64, pin_memory=True, drop_last=False)
val_cc_loader = DataLoader(dataset=valset, batch_size=args.batchsize, shuffle=False, num_workers=64, pin_memory=True, drop_last=False)

print('==> Start generating train boxes...')
trainset.generate_box = True

len_ds = len(trainset)
all_boxes_outer = update_box(train_cc_loader, model.module.encoder, 0.5)
assert len(all_boxes_outer) == len_ds
trainset.boxes_outer = all_boxes_outer.cpu()
    
trainset.generate_box = False
print('Train box generated!!!')

print('==> Start generating val boxes...')
valset.generate_box = True

len_ds = len(valset)
all_boxes_outer = update_box(val_cc_loader, model.module.encoder, 0.5)
assert len(all_boxes_outer) == len_ds
valset.boxes_outer = all_boxes_outer.cpu()

valset.generate_box = False
print('Val box generated!!!')
############################################

# meta-train loader
train_sampler = CategoriesSampler(trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=64, pin_memory=True)

val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=64, pin_memory=True)


if args.random_val_task == 0:
    print ('fix val set for all epochs')
    val_loader=[x for x in val_loader]

#label for query set, always in the same pattern
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)#012340123401234...
label = label.type(torch.LongTensor)
label = label.cuda()

optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

def save_model(name):
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))

trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0
writer = SummaryWriter(osp.join(args.save_path,'tf'))

result_list=[args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print (args.save_path)
    start_time=time.time()

    tl = Averager()
    ta = Averager()


    tqdm_gen = tqdm.tqdm(train_loader)
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm_gen, 1):

        global_count = global_count + 1
        data, _ = [_.cuda() for _ in batch]

        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        loss = F.cross_entropy(logits, label)

        acc = count_acc(logits, label)
        writer.add_scalar('data/loss', float(loss), global_count)
        writer.add_scalar('data/acc', float(acc), global_count)

        total_loss = loss/args.bs#batch of tasks, done by accumulate gradients
        writer.add_scalar('data/total_loss', float(total_loss), global_count)
        tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'
              .format(epoch, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()

        detect_grad_nan(model)
        if i%args.bs==0: #batch of tasks, done by accumulate gradients
            optimizer.step()
            optimizer.zero_grad()


    tl = tl.item()
    ta = ta.item()
    vl = Averager()
    va = Averager()

    #validation
    model.eval()
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.way * args.shot
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'meta'
            if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
            logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))

            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)

    vl = vl.item()
    va = va.item()
    writer.add_scalar('data/val_loss', float(vl), epoch)
    writer.add_scalar('data/val_acc', float(va), epoch)
    tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print('epoch: {}, val_loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    print ('val acc:%.4f'%va)
    if va >= trlog['max_acc']:
        print ('*********A better model is found*********')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    result_list.append('epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f'%(epoch,tl,ta,vl,va))

    torch.save(trlog, osp.join(args.save_path, 'trlog'))

    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print ('This epoch takes %d seconds'%(time.time()-start_time),'\nstill need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
    lr_scheduler.step()

writer.close()

# Test Phase
trlog = torch.load(osp.join(args.save_path, 'trlog'))
print('Test episodes: ', args.test_episode)
test_acc_record = np.zeros((args.test_episode,))
model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
model.eval()

# data prepare
testset = Dataset('test', args)
test_cc_loader = DataLoader(dataset=testset, batch_size=args.batchsize, shuffle=False, num_workers=64, pin_memory=True, drop_last=False)
########################################
print('==> Start generating test boxes...')
testset.generate_box = True

len_ds = len(testset)
all_boxes_outer = update_box(test_cc_loader, model.module.encoder, 0.5)
assert len(all_boxes_outer) == len_ds
testset.boxes_outer = all_boxes_outer.cpu()

testset.generate_box = False
print('Test box generated!!!')
########################################

sampler = CategoriesSampler(testset.label, args.test_episode, args.way, args.shot + args.query)
test_loader = DataLoader(testset, batch_sampler=sampler, num_workers=64, pin_memory=True)

ave_acc = Averager()
label = torch.arange(args.way).repeat(args.query)
if torch.cuda.is_available():
    label = label.type(torch.cuda.LongTensor)
else:
    label = label.type(torch.LongTensor)

tqdm_gen = tqdm.tqdm(test_loader)
with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        acc = count_acc(logits, label)* 100
        ave_acc.add(acc)
        test_acc_record[i-1] = acc
        tqdm_gen.set_description('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item(), acc))


m, pm = compute_confidence_interval(test_acc_record)

result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))

print(result_list[-2])
print(result_list[-1])

save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
