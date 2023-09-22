import argparse
import os
import wandb
from tqdm import tqdm
import numpy as np

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, save_yaml, calculate_PSNR


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
    model = get_model(opt['model'])
    if model.restored_epoch is not None and model.restored_step is not None:
        start_epoch = model.restored_epoch + \
                      (1 if (model.restored_step % (len(train_loader.dataset) // train_loader.batch_size)) == 0 else 0)
        step = model.restored_step + 1
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
                model.feed_data(batch)
                model.optimize_parameters()
                model.update_learning_rate(step)

                wandb.log({"tr_loss": model.loss.item(),
                           'step': step,
                           'epoch': epoch,
                           'learning_rate': model.get_current_learning_rate()})

                # validate
                if step == 1 or step % validation_freq == 0:
                    va_loss = []
                    va_psnr = []
                    for data in test_loader:
                        model.feed_data(data)
                        model.test()
                        va_loss.append(model.loss.item())
                        va_psnr.append(calculate_PSNR(model.gt_kernel.detach(),
                                                      model.pred_kernel.detach().squeeze(1), max_val='auto'))
                    wandb.log({'va_loss': np.mean(va_loss),
                               'va_psnr': np.mean(va_psnr),
                               'step': step,
                               'epoch': epoch})

                # save model
                if step == 1 or step % checkpoint_freq == 0:
                    model.save_network(os.path.join(experiment_path, f'{step}_network.pth'))
                    model.save_training_state(os.path.join(experiment_path, f'{step}_training_state.pth'), epoch, step)

                step += 1
                pbar.update(batch['lr'].shape[0])
                pbar.set_postfix({'loss (batch)': f'{model.loss.item():5.3f}',
                                  'lr': model.get_current_learning_rate()})


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
