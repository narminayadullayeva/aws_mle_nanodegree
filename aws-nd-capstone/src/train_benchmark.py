# Adapted from: https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN/blob/master/convolutional-neural-network/training.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
from PIL import Image

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ABID Counting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=10, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')
parser.add_argument('--save-model-id', default=1, type=int, metavar='N', help='id of the model to be saved')
parser.add_argument('--lastlayer', default=False, type=bool, metavar='BOOL', help='last layer or full network')
parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])



def show_image(im,target):
    print('count: %d' % target[0])
    npimg = np.transpose(im[0].mul(255).byte().numpy(), (1,2,0))
    img_batch = npimg
    for idx in range(5):
        print('count : %d' % target[idx+1])
        npimg = np.transpose(im[idx+1].mul(255).byte().numpy(), (1,2,0))
        img_batch = np.concatenate((img_batch,npimg), axis=0)
    img_pair_disp = Image.fromarray(img_batch)
    img_pair_disp.show()

best_prec = 0
train_loss_list = []
val_acc_list= []
train_acc_list = []
def main():
    global args, best_prec, train_loss_list, val_acc_list, train_acc_list
    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch]()

    
#     #training only the last layer
#     if args.lastlayer == True:
#         for param in net.parameters():
#             param.requires_grad = False
            
#     else:
#         for param in net.parameters():
#             param.requires_grad = True
        
        
    in_features = net.fc.in_features
    new_fc = nn.Linear(in_features,6)
    net.fc = new_fc

    net.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            train_loss_list = checkpoint['train_loss_list']
            val_acc_list = checkpoint['val_acc_list']
            train_acc_list = checkpoint['train_acc_list']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    params = net.parameters()
    snapshot_fname = "snapshots/%s_%s.pth.tar" % (args.arch, args.save_model_id)
    snapshot_best_fname = "snapshots/%s_%s_best.pth.tar" % (args.arch, args.save_model_id)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.train, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.test, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

#     optimizer = torch.optim.SGD(params, args.lr,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(params, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)


    # evaluate on validation set
    if args.evaluate == True:
        validate(val_loader, net, criterion, True)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch)

        # evaluate on validation set
        prec = validate(val_loader, net, criterion, False)

        # remember best prec@1 and save checkpoint
        # is_best = prec > best_prec
        # best_prec = max(prec, best_prec)
        # filename = "snapshots/%s_%s.pth.tar" % (args.arch, args.save_model_id)
        # torch.save({ 
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': net.state_dict(),
        #     'best_prec': best_prec,
        #     'train_loss_list': train_loss_list,
        #     'val_acc_list': val_acc_list,
        #     'train_acc_list': train_acc_list
        # }, snapshot_fname)
        # if is_best:
        #     shutil.copyfile(snapshot_fname,snapshot_best_fname)

def train(train_loader, net, criterion, optimizer, epoch):
    cur_lr = adjust_learning_rate(optimizer, epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        #show_image(input,target)

        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        score,idx = torch.max(output.data,1)
        correct = (target==idx)
        acc = float(correct.sum())/input.size(0)

        losses.update(loss.item(), input.size(0))
        train_acc.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}] lr {cur_lr:.5f}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec {train_acc.val:.3f} ({train_acc.avg:.3f})'.format(
               epoch, i, len(train_loader), cur_lr=cur_lr, batch_time=batch_time,
               data_time=data_time, loss=losses, train_acc=train_acc))
        train_loss_list.append(losses.val)
        
    train_acc_list.append(train_acc.avg)

def validate(val_loader, net, criterion, file_out):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()

    # switch to evaluate mode
    net.eval()
    
    if file_out==True:
      f = open('counting_result.txt','w') 

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            input = input.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = net(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            score,idx = torch.max(output.data,1)
            correct = (target==idx)
            acc = float(correct.sum())/input.size(0)

            if file_out==True:
              for j in range(input.size(0)):
                f.write('%d\n' % idx[j][0])

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            val_acc.update(acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {val_acc.val:.3f} ({val_acc.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   val_acc=val_acc))

    print(' * Prec {val_acc.avg:.3f}'.format(val_acc=val_acc))
    if file_out==True:
      f.close()

    val_acc_list.append(val_acc.avg)
    return val_acc.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lrd epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrd))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()