import argparse
import os
import wandb
from tqdm import tqdm
import numpy as np

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, save_yaml


def train(opt):
    # pass parameter
    project_name = opt['project_name']
    experiment_path = os.path.join('./experiments', opt['experiment_name'])
    max_epochs = opt['training']['max_epochs']
    validation_freq = opt['training']['validation_freq']
    checkpoint_freq = opt['training']['checkpoint_freq']
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
    F_model = get_model(opt['F_Model'])
    U_model = get_model(opt['U_Model'])
    if F_model.restored_epoch is not None and F_model.restored_step is not None:
        start_epoch = F_model.restored_epoch + \
                      (1 if (F_model.restored_step % (
                                  len(train_loader.dataset) // train_loader.batch_size)) == 0 else 0)
        step = F_model.restored_step + 1
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
                U_model.feed_data(batch)
                U_model.test()
                pred_kernel = U_model.pred_kernel.detach().cpu()
                F_model.feed_data({'lr': batch['lr'],
                                   'hr': batch['hr'],
                                   'kernel': pred_kernel.squeeze(1)})
                F_model.optimize_parameters()
                F_model.update_learning_rate(step)

                wandb.log({"tr_loss": F_model.loss.item(),
                           'step': step,
                           'epoch': epoch,
                           'learning_rate': F_model.get_current_learning_rate()})

                # validate
                if step == 1 or step % validation_freq == 0:
                    va_loss = []
                    for data in test_loader:
                        U_model.feed_data(data)
                        U_model.test()
                        va_pred_kernel = U_model.pred_kernel.detach().cpu()
                        F_model.feed_data({'lr': data['lr'],
                                           'hr': data['hr'],
                                           'kernel': va_pred_kernel.squeeze(1)})
                        F_model.test()
                        va_loss.append(F_model.loss.item())
                    wandb.log({"va_loss": np.mean(va_loss),
                               'step': step,
                               'epoch': epoch})

                # save model
                if step == 1 or step % checkpoint_freq == 0:
                    F_model.save_network(os.path.join(experiment_path, f'{step}_network.pth'))
                    F_model.save_training_state(os.path.join(experiment_path, f'{step}_training_state.pth'), epoch,
                                                step)

                step += 1
                pbar.update(batch['lr'].shape[0])
                pbar.set_postfix({'loss (batch)': f'{F_model.loss.item():5.3f}',
                                  'lr': F_model.get_current_learning_rate()})


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
