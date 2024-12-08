import torch
from .util.util import AverageMeter, clip_gradients, cancel_gradients_last_layer
from torch.utils.tensorboard import SummaryWriter


def train_on_epoch(
    student, 
    teacher, 
    train_loader, 
    dino_loss, 
    epoch,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    use_amp=False,
    clip_grad=3.,
    freeze_last_layer=1,
    log_step=10,
    log_writer=None,
):
    batch_loss = AverageMeter()
    writer = SummaryWriter() if log_writer is not None else None

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for batch, (images, _) in enumerate(train_loader):
        batch = len(train_loader) * epoch + batch
        
        ############ schedule learning rate and weight decay ############
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[batch]
            if i == 0:
                param_group['weight_decay'] = wd_schedule[batch]
        #################################################################

        images = [img.cuda(non_blocking=True) for img in images]

        if use_amp:
            with torch.cuda.amp.autocast():
                teacher_output = teacher(images[:2])
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)

        else:
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
        
        batch_loss.update(loss, images[0].shape[0])

        ############ update optimizer and backpropagate loss ############
        optimizer.zero_grad()
        if use_amp:
            loss.backward()
            if clip_grad:
                param_norms = clip_gradients(student, clip_grad)
            cancel_gradients_last_layer(epoch, student, freeze_last_layer)
            optimizer.step()
        
        else:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                param_norms = clip_gradients(student, clip_grad)
            cancel_gradients_last_layer(epoch, student, freeze_last_layer)
            scaler.step(optimizer)
            scaler.update()
        ##################################################################

        ################### EMA update for the teacher ###################
        with torch.no_grad():
            m = momentum_schedule[batch]
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        ##################################################################

        ####### write logs and hyperparameters by cosine scheduler #######
        if writer is not None:
            writer.add_scalar('Train/loss', loss, batch)
            writer.add_scalar('hyperparameters/lr', optimizer.param_groups[0]["lr"], batch)
            writer.add_scalar('hyperparameters/weight_decay', optimizer.param_group["weight_decay"], batch)
            writer.add_scalar('hyperparameters/momentum', optimizer.param_gruops[0]["momentum"], batch)
        ##################################################################

        if log_step % batch == 0:
            print(f'\n{" "*10} [Train Batch {batch+1}/{len(train_loader)}] {" "*10}'
                  f'\ntrain loss: {loss:.3f}')

    return {
        'train_loss': batch_loss.avg,
    }


def validate_on_epoch(
    student,
    teacher,
    validation_loader,
    dino_loss,
    epoch,
    use_amp,
    log_step,
    log_writer=None,
):
    batch_loss = AverageMeter()
    writer = SummaryWriter if log_writer is not None else None
    
    for batch, (images, _) in enumerate(validation_loader):
        batch = len(validation_loader) * epoch + batch

        images = [img.cuda(non_blocking=True) for img in images]

        if use_amp:
            with torch.cuda.amp.autocast():
                teacher_output = teacher(images[:2])
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)

        else:
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
        
        batch_loss.update(loss, images[0].shape[0])

        if log_step % batch == 0:
            print(f'\n{" "*10} [Validate Batch {batch+1}/{len(validation_loader)}] {" "*10}'
                  f'\nvalid loss: {loss:.3f}')

        if writer is not None:
            writer.add_scalar('Valid/loss', loss, batch)

    return {
        'val_loss': batch_loss.avg,
    }