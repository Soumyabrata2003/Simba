#!/usr/bin/env python
from __future__ import print_function
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from cosine_lr_schedueler import CosineLRScheduler
from torch.optim.lr_scheduler import MultiStepLR
# from utils.solver import _lr_scheduler
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from utils.utils import gen_label
from utils.NCELoss import NCELoss, LabelSmoothingCrossEntropy
# from torchlight import DictAction
import resource
import copy
from torch import linalg as LA
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# from text_prompt import text_prompt
# torch.multiprocessing.set_sharing_strategy('file_system')

"https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py"


def ema_update(source, target, decay=0.99, start_itr=20, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr<start_itr:
        decay = 0.0
    # source = copy.deepcopy(source)
    with torch.no_grad():
        # for key, value in source.module.state_dict().items():
        for key, value in source.state_dict().items():
            target.state_dict()[key].copy_(target.state_dict()[key] * decay + value * (1 - decay))
            # print(key)


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = False

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]

def epoch_saving(epoch, model, optimizer, filename):
    state_dict = model.state_dict()
    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': weights,
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename
    
def epoch_saving_2(epoch, model, optimizer, lr_scheduler, filename):
    state_dict = model.state_dict()
    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': weights,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_states': lr_scheduler.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():


    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='GRA Transformer')

    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # gra
    parser.add_argument(
        '--joint-label',
        type=list,
        default=[],
        help='tells which group each joint belongs to')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=2, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=10,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--ema',
        # action="store_true",
        default=False,
        help='ema weight for eval')

    parser.add_argument('--lambda_1', type=float, default=1e-4)
    parser.add_argument('--lambda_2', type=float, default=1e-1)

    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=48,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        # action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        # action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        # action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--resume',
        default=None,
        help='the resume checkpoint for network')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[110, 120],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=[0,1],
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--lr-type', default='cosine', help='type of lr_scheduler')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='nesterov momentum')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
        self.model = self.model.cuda(self.output_device)
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            # self.load_lr_scheduler()
            self.load_data()
            
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.best_acc_ema = 0
        self.best_acc_epoch_ema = 0

        # self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
        if self.arg.ema:
            Model = import_class(self.arg.model)
            self.model_ema = Model(**self.arg.model_args).cuda(self.output_device)
            ema_update(self.model, self.model_ema, itr=0)


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                pin_memory=True,
                prefetch_factor=16,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        # self.model = self.model.double()
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.loss2 = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            # weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        
        if self.arg.resume:
            self.global_step = int(self.arg.resume[:-3].split('-')[-1])
            self.print_log('Resume weights from {}.'.format(self.arg.resume))
            if '.pkl' in self.arg.resume:
                with open(self.arg.resume, 'r') as f:
                    checkpoint = pickle.load(f)
            else:
                checkpoint = torch.load(self.arg.resume, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            weights=checkpoint['model_state_dict']
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            
            self.arg.start_epoch = checkpoint['epoch'] + 1
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            # my_list = ['fc.weight', 'fc.bias', 'fc_mu.weight', 'fc_mu.bias','fc_logvar.weight', 'fc_logvar.bias', 'decoder.weight', 'decoder.bias']
            # params = list(filter(lambda kv: kv[0] in my_list, self.model.named_parameters()))
            # base_params = list(filter(lambda kv: kv[0] not in my_list, self.model.named_parameters()))
            #
            # params = [i[1] for i in params]
            # base_params = [i[1] for i in base_params]

            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.momentum,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'NAdam':
            self.optimizer = optim.NAdam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
        if self.arg.resume:
            checkpoint = torch.load(self.arg.resume, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.print_log('Loaded optimizer from epoch: {} !'.format(self.arg.start_epoch -1))

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def load_lr_scheduler(self):
        if self.arg.lr_type == "multistep":
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
            if self.arg.resume is not None:
                scheduler_states = torch.load(self.arg.resume)['lr_scheduler_states']
                self.print_log(f'Loading LR scheduler states from: {self.arg.resume}')
                self.lr_scheduler.load_state_dict(scheduler_states)
                self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
                self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')
        
        if self.arg.lr_type == "cosine":
            self.lr_scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            self.arg.num_epoch,
            warmup_epochs=self.arg.warm_up_epoch
            )
            if self.arg.resume is not None:
                scheduler_states = torch.load(self.arg.resume)['lr_scheduler_states']
                self.print_log(f'Loading LR scheduler states from: {self.arg.resume}')
                self.lr_scheduler.load_state_dict(scheduler_states)
                self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
                self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')
    
        

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' or self.arg.optimizer == 'NAdam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        if self.arg.ema:
            self.model_ema.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        print('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        loss_value2 = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        # mix_precision is slower for this model!!!
        use_amp = False #True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # torch.autograd.set_detect_anomaly(True)

        # texts = classes # n_cls 77

        soft_label_emma = 0
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                # data = data.double().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            
            # if (batch_idx + 1) == 1 or (batch_idx + 1) % 10 == 0:
            #     self.lr_scheduler.step(epoch + batch_idx / len(process)) #for warm_cosine_annealing_lr

            # print(data[0,:,0,:,0])

            # with autograd.detect_anomaly():
            # forward

            class SoftTargetCrossEntropy(nn.Module):

                def __init__(self):
                    super(SoftTargetCrossEntropy, self).__init__()

                def forward(self, x, target):
                    # because p has already been passed through softmax !
                    # cross entropy is non-symetric!! the order matters!!!
                    loss = -torch.sum(target * torch.log(x+1e-20), dim=-1)
                    return loss.mean()
            # print(label)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # output, z = self.model(data, F.one_hot(label, num_classes=self.model.module.num_class))
                # texts = texts.to('cuda')
                # label = label.to('cuda')
                # texts = texts[label]  # bs 77
                # texts = texts.to(torch.float32)
                output, z  = self.model(data, F.one_hot(label, num_classes=self.model.num_class))
                # print(output.shape,z.shape,label.shape)
                # ground_truth = torch.tensor(gen_label(label),dtype=output.dtype,device=output.device)
                # # gt = [bs bs]
                # # print(texts.device,label.device,output.device)
                # texts = texts.to(output.device)
                # label = label.to(output.device)
                # texts = texts[label]  # bs 77
                # texts = texts.to(torch.float32)
                # cls_emb = nn.Linear(77,60,dtype=torch.float32).to(output.device)(texts) #bs 60
                # ske_emb = output / output.norm(dim=-1, keepdim=True)
                # cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
                # logits = ske_emb @ cls_emb.t()  
                # print(output.shape,ground_truth.shape)
                # loss = criterion(logits, ground_truth)
                # mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, label, 60)
                loss = self.loss(output, label) #+ self.loss2(cls_emb,label)
                # lambda_1=1e-4
                # lambda_2=1e-1
                # loss = lambda_2* mmd_loss + lambda_1* l2_z_mean + loss
            loss2 = torch.zeros_like(loss).cuda(loss.device)



            loss += loss2


            # backward
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            loss_value2.append(loss2.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss2', loss2.data.item(), self.global_step)
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}. loss2: {:.4f}. Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(loss_value2), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            # state_dict = self.model.state_dict()
            # weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
            filename= self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt'
            # epoch_saving(epoch, self.model.module, self.optimizer, filename)
            epoch_saving(epoch, self.model, self.optimizer, filename)
            # epoch_saving_2(epoch, self.model, self.optimizer, self.lr_scheduler, filename)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        if self.arg.ema:
            self.model_ema.eval()
        # texts = classes # n_cls 77
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []

            loss_value_ema = []
            score_frag_ema = []
            label_list_ema = []
            pred_list_ema = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                if arg.ema:
                    label_list_ema.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    # data = data.double().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    # output, z = self.model(data, F.one_hot(label, num_classes=self.model.module.num_class))
                    # output, z = self.model(data, F.one_hot(label, num_classes=self.model.num_class))
                    # texts = texts.to('cuda')
                    # label = label.to('cuda')
                    # texts = texts[label]  # bs 77
                    # texts = texts.to(torch.float32)
                    output, z = self.model(data, F.one_hot(label, num_classes=self.model.num_class))
                    if arg.ema:
                        self.model_ema.cuda(self.output_device)
                        output_ema, z_ema = self.model_ema(data, F.one_hot(label, num_classes=self.model.num_class))

                    loss = self.loss(output, label)
                    # mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, label, 60)
                    # lambda_1=1e-4
                    # lambda_2=1e-1
                    # loss = lambda_2*mmd_loss + lambda_1*l2_z_mean + loss
                    # texts = texts.to(output.device)
                    # label = label.to(output.device)
                    # texts = texts[label]  # bs 77
                    # texts = texts.to(torch.float32)
                    # cls_emb = nn.Linear(77,60,dtype=torch.float32).to(output.device)(texts) #bs 60
                    # ske_emb = output / output.norm(dim=-1, keepdim=True)
                    # cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
                    # loss = self.loss(output,label) + self.loss2(cls_emb,label)
                    # print(ske_emb.dtype,cls_emb.dtype,output.dtype)
                    # logits = ske_emb @ cls_emb.t()  
                    # ground_truth = torch.tensor(gen_label(label),dtype=output.dtype,device=output.device)
                    # # gt = [bs bs]
                    # loss = criterion(logits, ground_truth)

                    if arg.ema:
                        loss_ema = self.loss(output_ema, label)

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    if arg.ema:
                        score_frag_ema.append(output_ema.data.cpu().numpy())
                        loss_value_ema.append(loss_ema.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

                    if arg.ema:
                        _, predict_label_ema = torch.max(output_ema.data, 1)
                        pred_list_ema.append(predict_label_ema.data.cpu().numpy())

                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            if self.arg.ema:
                score_ema = np.concatenate(score_frag_ema)
                loss_ema = np.mean(loss_value_ema)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            if self.arg.ema:
                accuracy_ema = self.data_loader[ln].dataset.top_k(score_ema, 1)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            if self.arg.ema:
                if accuracy_ema > self.best_acc_ema:
                    self.best_acc_ema = accuracy_ema
                    self.best_acc_epoch_ema = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.ema:
                print('Accuracy_ema: ', accuracy, ' model_ema: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
                if arg.ema:
                    self.val_writer.add_scalar('loss_ema', loss_ema, self.global_step)
                    self.val_writer.add_scalar('acc_ema', accuracy_ema, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            if self.arg.ema:
                score_dict_ema = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score_ema))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            if self.arg.ema:
                self.print_log('\tMean {} loss_ema of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value_ema)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            if arg.ema:
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}_ema: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score_ema, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            if arg.ema:
                if save_score:
                    with open('{}/epoch{}_{}_score_ema.pkl'.format(
                            self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                        pickle.dump(score_dict_ema, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

            if arg.ema:
                # acc for each class:
                label_list_ema = np.concatenate(label_list_ema)
                pred_list_ema = np.concatenate(pred_list_ema)
                confusion_ema = confusion_matrix(label_list_ema, pred_list_ema)
                list_diag_ema = np.diag(confusion_ema)
                list_raw_sum_ema = np.sum(confusion_ema, axis=1)
                each_acc_ema = list_diag_ema / list_raw_sum_ema
                with open('{}/epoch{}_{}_each_class_acc_ema.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(each_acc_ema)
                    writer.writerows(confusion_ema)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                # self.lr_scheduler.step(epoch)
                self.train(epoch, save_model=save_model)
                if self.arg.ema:
                    ema_update(self.model, self.model_ema, itr=epoch)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            # print(weights_path)
            checkpoint = torch.load(weights_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
            weights=checkpoint['model_state_dict']
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)] for k, v in weights.items()])

            # weights = torch.load(weights_path)
            # if type(self.arg.device) is list:
            #     if len(self.arg.device) > 1:
            #         weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            # mask = self.model.module.joint_label
            #
            # A = torch.tensor(self.model.module.graph.A).cuda(mask.device).float()
            # A[A!=0] = 1
            #
            # ind = torch.argmax(mask, dim=0)
            # print(ind)


            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    # classes = text_prompt()
    # n_class = classes.size(0)
    # criterion = NCELoss()
    processor.start()
