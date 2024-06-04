import argparse
import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, save_yaml, calculate_PSNR, PCA_Decoder


def train(opt):
    # pass parameter
    project_name = opt['project_name']
    experiment_path = os.path.join('./experiments', opt['experiment_name'])
    max_epochs = opt['training']['max_epochs']
    validation_freq = opt['training']['validation_freq']
    checkpoint_freq = opt['training']['checkpoint_freq']
    correct_step = opt['training']['correct_step']
    wandb_id = opt['training']['wandb_id']

    # mkdir, back up options
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    n = 0
    while True:
        opt_name = 'option.yaml' if n == 0 else f'option{n}.yaml'
        if os.path.exists(os.path.join(experiment_path, opt_name)):
            n += 1
        else:
            save_yaml(opt, os.path.join(experiment_path, opt_name))
            break

    # set up data loader
    train_loader = get_dataloader(opt['train_data'])
    test_loader = get_dataloader(opt['test_data'])

    # set up model, epoch, step
    F_model = get_model(opt['F_model'])
    P_model = get_model(opt['P_model'])
    C_model = get_model(opt['C_model'])
    if P_model.restored_epoch is not None and P_model.restored_step is not None:
        start_epoch = P_model.restored_epoch + (1 if (P_model.restored_step % (len(train_loader.dataset)
                                                                               // train_loader.batch_size)) == 0 else 0)
        step = P_model.restored_step + 1
    else:
        start_epoch, step = 1, 1

    # set up wandb
    wandb.init(project=project_name,
               name=opt['experiment_name'],
               resume='allow',
               id=wandb_id)

    # start training
    for epoch in range(start_epoch, max_epochs + 1):
        with tqdm(desc=f'Epoch {epoch}/{max_epochs}', total=len(train_loader.dataset), unit='img') as pbar:
            for batch in train_loader:
                P_model.feed_data(batch)
                P_model.optimize_parameters()
                P_model.update_learning_rate(step)
                kernel_code_of_sr = P_model.pred_kernel_code.detach().cpu()
                gt_kernel_code = F_model.pca_encoder(batch['kernel'].to(F_model.device))

                for _ in range(correct_step):
                    F_model.feed_data({'hr': batch['hr'],
                                       'lr': batch['lr'],
                                       'kernel_code': kernel_code_of_sr})
                    F_model.test()
                    C_model.feed_data({'sr': F_model.sr.detach().cpu(),
                                       'kernel_code_of_sr': kernel_code_of_sr,
                                       'gt_kernel_code': gt_kernel_code})
                    C_model.optimize_parameters()
                    kernel_code_of_sr = C_model.pred_kernel_code.detach().cpu()
                C_model.update_learning_rate(step)

                wandb.log({'tr_P_loss': P_model.loss.item(),
                           'tr_F_loss': F_model.loss.item(),
                           'tr_C_loss': C_model.loss.item(),
                           'step': step,
                           'epoch': epoch,
                           'P_learning_rate': P_model.get_current_learning_rate(),
                           'C_learning_rate': C_model.get_current_learning_rate()})

                # validate
                if step == 1 or step % validation_freq == 0:
                    va_P_loss = []
                    va_F_loss = []
                    va_C_loss = []
                    va_sr_psnr = []
                    va_kernel_psnr = []
                    pca_decoder = PCA_Decoder(weight=F_model.pca_encoder.weight, mean=F_model.pca_encoder.mean)
                    for data in test_loader:
                        P_model.feed_data(data)
                        P_model.test()
                        kernel_code_of_sr = P_model.pred_kernel_code.detach().cpu()
                        gt_kernel_code = F_model.pca_encoder(data['kernel'].to(F_model.device))
                        va_P_loss.append(P_model.loss.item())
                        for _ in range(correct_step):
                            F_model.feed_data({'hr': data['hr'],
                                               'lr': data['lr'],
                                               'kernel_code': kernel_code_of_sr})
                            F_model.test()
                            sr = F_model.sr.detach().cpu()
                            C_model.feed_data({'sr': sr,
                                               'kernel_code_of_sr': kernel_code_of_sr,
                                               'gt_kernel_code': gt_kernel_code})
                            C_model.test()
                            kernel_code_of_sr = C_model.pred_kernel_code.detach().cpu()
                        va_F_loss.append(F_model.loss.item())
                        va_C_loss.append(C_model.loss.item())
                        va_sr_psnr.append(calculate_PSNR(F_model.hr.detach(), F_model.sr.detach(), max_val=1.0))
                        pred_kernel = pca_decoder(kernel_code_of_sr.to(F_model.device))
                        va_kernel_psnr.append(calculate_PSNR(pred_kernel,
                                                             data['kernel'].to(F_model.device), max_val='auto'))
                    wandb.log({'va_P_loss': np.mean(va_P_loss),
                               'va_F_loss': np.mean(va_F_loss),
                               'va_C_loss': np.mean(va_C_loss),
                               'va_sr_psnr': np.mean(va_sr_psnr),
                               'va_kernel_psnr': np.mean(va_kernel_psnr),
                               'step': step,
                               'epoch': epoch})

                # save model
                if step == 1 or step % checkpoint_freq == 0:
                    P_model.save_network(os.path.join(experiment_path, f'{step}_P_network.pth'))
                    P_model.save_training_state(os.path.join(experiment_path, f'{step}_P_training_state.pth'), epoch,
                                                step)
                    C_model.save_network(os.path.join(experiment_path, f'{step}_C_network.pth'))
                    C_model.save_training_state(os.path.join(experiment_path, f'{step}_C_training_state.pth'), epoch,
                                                step)

                step += 1
                pbar.update(batch['lr'].shape[0])
                pbar.set_postfix({'P_loss (batch)': f'{P_model.loss.item():5.3f}',
                                  'C_loss (batch)': f'{C_model.loss.item():5.3f}',
                                  'P_lr': P_model.get_current_learning_rate(),
                                  'C_lr': C_model.get_current_learning_rate()})


def main():
    """please make sure that the pwd is .../PsfPred rather than .../PsfPred/codes/trains"""
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/train_something.yaml')
    args = prog.parse_args()

    # start train
    train(read_yaml(args.opt))


if __name__ == '__main__':
    main()
