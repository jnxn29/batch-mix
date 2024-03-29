# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.mix1_modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
import time
import csv
from torch import nn

logger = logging.getLogger(__name__)

def mixt(x, lam, num_merge=2):
    batch_size = x.size(0)
    merge_size = torch.div(batch_size, num_merge, rounding_mode='trunc')#

    # 使用 torch.chunk 将 x 切分为 num_merge 部分
    chunks = torch.split(x, split_size_or_sections=merge_size, dim=0)

    # 选择第一个 chunk
    result_x = chunks[0]

    # 对后续的 chunk 进行混合
    for i in range(1, num_merge):
        result_x = lam * result_x + (1 - lam) * chunks[i]

    return result_x
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100   #种类设置

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step):
    # Validation!
    print(f"Entering valid function. global_step = {global_step}")  # 添加调试输出
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            outputs = model(x)
            logits = outputs[0]  # 假设logits是tuple的第一个元素

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    print(f"Leaving valid function. global_step = {global_step}, accuracy = {accuracy}")  # 添加调试输出

    return accuracy, eval_losses.avg


def train(args, model):
    """ Train the model """
    #此处创建文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mix0.80.8_{args.dataset}lr{args.learning_rate}_eval_every{args.eval_every}_{timestamp}_3080.csv"
    if args.fp16:
        filename = f"mix0.80.8_{args.dataset}lr{args.learning_rate}_eval_every{args.eval_every}fp16_{timestamp}_3080_.csv"
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']  # 列名
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        print("train_loader", len(train_loader))
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        start = time.time()  # 开始时间
        train_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            num_classes = 10 if args.dataset == "cifar10" else 100  # 种类设置
            y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
            # print(" y_onehot",y_onehot)
            # outputs = model(x)
            # # print("outputs",outputs)
            # if isinstance(outputs, tuple):
            #     outputs = outputs[0]  # 假设模型返回的是一个元组,取第一个元素作为输出
            #
            # loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, y_onehot)

            ##bm
            # BM方法: 对输入数据和标签进行混合
            lam = np.random.beta(0.8, 0.8)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(args.device)

            mixed_y = mixt(y_onehot, lam,2)
            # print("mixed_y",mixed_y.size())

            outputs = model(x,lam=lam,merge=2)
            if isinstance(outputs, torch.Tensor):
                loss = outputs
                # print("loss:", loss.item())
            else:
                logits, attn_weights = outputs
                # print("logits shape:", logits.shape)
                # print("attn_weights shape:", [weights.shape for weights in attn_weights])

                # 计算损失
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_classes), mixed_y.argmax(dim=1).view(-1))
                # print("loss:", loss.item())
            ##bm

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
            train_loss += loss.item()

            if global_step % t_total == 0:
                break
            losses.reset()
            # print(global_step % args.eval_every)
            if global_step % args.eval_every == 0 :
                finish = time.time()  # 结束时间
                epoch_time = finish - start  # 训练时间
                accuracy,test_loss = valid(args, model, test_loader, global_step)
                lr = optimizer.param_groups[0]['lr']  # 学习率
                print('Steps {} training time consumed: {:.2f}s'.format(global_step, epoch_time))  # 打印训练时间
                print('Learning rate: {:.6f}'.format(lr))  # 打印学习率
                print('Accuracy: {:.2f}'.format(accuracy))  # 打印准确率
                num_epochs = t_total // len(train_loader)
                print('Epoch{} training time consumed: {:.2f}s'.format(global_step // len(train_loader), epoch_time))
                text1 = {
                    'epoch': global_step / len(train_loader),
                    'train loss': train_loss / len(train_loader),
                    'lr': lr,
                    'test loss': test_loss,
                    'test acc': accuracy,
                    'time': epoch_time
                }
                fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']  # 列名
                reordered_dict = {key: text1[key] for key in fieldnames}
                with open(filename, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(reordered_dict)

                start= time.time()
                # 保存数据
                model.train()
            if global_step > t_total :
                return
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100","imagenet"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=1563*50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
