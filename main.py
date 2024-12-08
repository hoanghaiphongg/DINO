import os
import time
import argparse
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torchsummary import summary

from .util.dataset import get_dataset
from .util.loss import DINOLoss
from .util.util import get_params_groups, cosine_scheduler
from .util.callback import CheckPoint
from .models.model_utils import MultiCropWrapper, DINOHead
from .models.resnet import ResNet50
from .models import vit
from .train import train_on_epoch, validate_on_epoch


def fit(student, teacher, train_loader, validation_loader, dino_loss, 
        epochs, optimizer, lr_schedule, wd_schedule, momentum_schedule, 
        use_amp, clip_grad, freeze_last_layer, check_point, 
        train_log_step=300, valid_log_step=100, log_writer=True):
    
    if check_point:
        cp = CheckPoint(verbose=True)

    print('Start Model Training...!')
    start_training = time.time()
    pbar = tqdm(range(epochs), total=int(epochs))
    for epoch in pbar:

        start_epoch = time.time()

        train_stats = train_on_epoch(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            dino_loss=dino_loss,
            epoch=epoch,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            use_amp=use_amp,
            clip_grad=clip_grad,
            freeze_last_layer=freeze_last_layer,
            log_step=train_log_step,
            log_writer=log_writer,
        )

        train_loss = train_stats['train_loss']

        valid_stats = validate_on_epoch(
            student=student,
            teacher=teacher,
            validation_loader=validation_loader,
            dino_loss=dino_loss,
            epoch=epoch,
            use_amp=use_amp,
            log_step=valid_log_step,
            log_writer=log_writer,
        )

        valid_loss = valid_stats['val_loss']

        end_epoch = time.time()

        print(f'\n{"="*40} Epoch {epoch+1}/{epochs} {"="*40}'
              f'\ntime : {end_epoch-start_epoch:.2f}s'
              f'   train average loss : {train_loss:.3f}'
              f'\n   valid average loss : {valid_loss:.3f}')
        print(f'\n{"="*100}')

        if check_point:
            os.makedirs('./weights', exist_ok=True)
            cp(valid_loss, teacher, f'./weights/teacher_{epoch}.pt')
            cp(valid_loss, student, f'./weights/student_{epoch}.pt')

    end_training = time.time()
    print(f'Total Training Time: {end_training-start_training:.2f}s')

    return {
        'teacher': teacher,
        'student': student,
    }


def get_args_parser():
    parser = argparse.ArgumentParser(description='DINO', add_help=False)

    # hyperparameters and training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='a batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial value of learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='minimum value of learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='initial value of weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='maximum value of weight decay for cosine scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=10,
                        help='number of epochs for warm-up scheduling')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='optimizer for training model')
    parser.add_argument('--use_amp', type=bool, default=True,
                        help='using half precision float on training')
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help='gradient clipping')
    parser.add_argument('--freeze_last_layer', type=int, default=1,
                        help='freeze last layer of model')
    parser.add_argument('--check_point', type=bool, default=True,
                        help='save weights of each models')
    parser.add_argument('--train_log_step', type=int, default=300,
                        help='print log of training')
    parser.add_argument('--valid_log_step', type=int, default=100,
                        help='print log of validating')
    parser.add_argument('--log_writer', type=bool, default=True,
                        help='write log of training and validating in tensorboard')
    
    # model paramters
    parser.add_argument('--model', type=str, default='vit_t',
                        choices=['resnet', 'vit_t', 'vit_s', 'vit_b'],
                        help='a name of vision transforer or resnet')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch size for vision transformer model')
    parser.add_argument('--out_dim', type=int, default=65536,
                        help='dimension of the DINO head output. For complex and large datasets large values work well')
    parser.add_argument('--use_bn_in_head', type=bool, default=False,
                        help='batch normalization in projection head')
    parser.add_argument('--norm_last_layer', type=int, default=True,
                        help='Whether or not to weight normalize the last layer of the DINO head')
    parser.add_argument('--momentum_teacher', type=float, default=0.996,
                        help='Base EMA parameter for teacher update and the value is increased to 1 during training with cosine schedule.')

    # temperature paramters
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.04,
                        help='initial value for the teacher temperature')
    parser.add_argument('--teacher_temp', type=float, default=0.04,
                        help='Final value (after linear warmup) of the teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', type=float, default=0,
                        help='number of warmup epochs for the teacher temperature')

    # augmentation parameters
    parser.add_argument('--global_img_size', type=int, default=224,
                        help='image size of global view in augmentations')
    parser.add_argument('--local_img_size', type=int, default=96, 
                        help='image size of local view in augmentations')
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help='crop scale for global view in augmentations')
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help='crop scale for local view in augmentations')
    parser.add_argument('--local_crops_number', type=float, default=8,
                        help='the number of multi cropping for training')
    
    # etc paramters
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='set device for faster model training')
    return parser


def main(args):
    device = torch.device(args.device)
    print(f'device is {args.device}...')

    dataset = get_dataset(
        global_img_size=args.global_img_size,
        local_img_size=args.local_img_size,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
        batch_size=args.batch_size,
    )

    train_loader, train_len = dataset['train'], dataset['number_of_train']
    valid_loader, valid_len = dataset['valid'], dataset['number_of_valid']

    print(f'train {train_len}, valid {valid_len} data ready...')

    if 'resnet' in args.model:
        student = ResNet50()
        teacher = ResNet50()

    elif 'vit' in args.model:
        if args.model == 'vit_t':
            student = vit.vit_tiny(patch_size=args.patch_size)
            teacher = vit.vit_tiny(patch_size=args.patch_size)

        elif args.model == 'vit_s':
            student = vit.vit_small(patch_size=args.patch_size)
            teacher = vit.vit_small(patch_size=args.patch_size)
        
        elif args.model == 'vit_b':
            student = vit.vit_base(patch_size=args.patch_size)
            teacher = vit.vit_base(patch_size=args.patch_size)
        
        else:
            raise ValueError(f'{args.model} does not exist, choose between vit_t, vit_s or vit_b')

    else:
        raise ValueError(f'{args.model} does not exist, choose vit or resnet')

    embed_dim = student.embed_dim
    student = MultiCropWrapper(
        student,
        DINOHead(
            in_dim=embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
    ).to(device)

    teacher = MultiCropWrapper(
        teacher,
        DINOHead(
            in_dim=embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
        )
    ).to(device)
    
    print(f'{args.model} student and teacher networks ready...')
    # summary(student, (3, args.global_img_size, args.global_img_size), device='cpu')

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=args.local_crops_number + 2,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        epochs=args.epochs,
    ).to(device)

    params_groups = get_params_groups(student)
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(params_groups)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params_groups, lr=0, momentum=0.9)
    else:
        raise ValueError(f'{args.optimizer} does not exist')
    
    print(f'dino loss and optimizer {args.optimizer} ready...')

    lr_schedule = cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epoch,
    )

    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_epoch=len(train_loader),
    )

    momentum_schedule = cosine_scheduler(
        base_value=args.momentum_teacher,
        final_value=1,
        epochs=args.epochs,
        niter_per_epoch=len(train_loader),
    )
    print('schedulers of learning rate, weight decay and momentum ready')

    history = fit(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        validation_loader=valid_loader,
        dino_loss=dino_loss,
        epochs=args.epochs,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        wd_schedule=wd_schedule,
        momentum_schedule=momentum_schedule,
        use_amp=args.use_amp,
        clip_grad=args.clip_grad,
        freeze_last_layer=args.freeze_last_layer,
        check_point=args.check_point,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
        log_writer=args.log_writer,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)