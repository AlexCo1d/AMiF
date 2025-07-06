# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and
    DInterface can be seen as transparent to all your args.
"""
import os

from argparse import ArgumentParser
import lightning.pytorch as pl

import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import WandbLogger
from torch import distributed as dist

from dataset.MM_VQA_datamodule import VQADataModule
from dataset.MM_pretrain_datamodule import PretrainDataModule
from model.MM_VQA import MM_VQA
from model.MM_pretrain import MM_pretrain
from model.biomedclip_val import BiomedCLIP


def main():
    seed_everything(42, workers=True)
    config = yaml.safe_load(open('./Config/config_VQA.yaml', 'r'))
    if config['num_gpus'] != len(config['device']):
        config['num_gpus'] = len(config['device'])

    save_path = config['save_path']
    os.makedirs(save_path, exist_ok=True)

    callbacks = [plc.LearningRateMonitor(logging_interval='step'),
                 # plc.ModelCheckpoint(save_top_k=-1, dirpath=save_path, filename='epoch-{epoch}',
                 #                     save_on_train_epoch_end=True,
                 #                     save_last=True, every_n_epochs=2)
                 plc.ModelCheckpoint(monitor=config['valid']['monitor'], dirpath=os.path.join(save_path, 'ckpt'),
                                     filename='{epoch}-{step}-{'+config['valid']['monitor']+':.4f}', save_last=True,
                                     mode="max", save_top_k=20),
                 # plc.EarlyStopping(monitor="val_metric",
                 #                   patience=10, verbose=True, mode="max")
                 ]

    logger_path = os.path.join(save_path, 'logs')
    os.makedirs(logger_path, exist_ok=True)

    wandb_logger = WandbLogger(
        project=f"MM_{config['dataset']}", save_dir=logger_path, name='Pretrain')

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        devices=config['device'],
        precision=config['precision'],
        benchmark=True,
        accumulate_grad_batches=config['accumulate_grad_steps'],
        log_every_n_steps=config['log_freq'],
        val_check_interval=config['valid']['val_step_freq'],
    )


    # load datamodule
    dm = VQADataModule(config)

    # load the model
    model = MM_VQA(config, dataloader=dm.train_dataloader())
    model.training_steps = model.num_training_steps(trainer, dm)
    print('train dataset size:', len(dm.train_dataloader().dataset))
    print('test dataset size:', len(dm.val_dataloader().dataset))

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader(),
                ckpt_path=config['resume_checkpoint'] if config['resume_checkpoint'] != "" else None)

    # model = BiomedCLIP(config)
    # trainer.validate(model, dm.val_dataloader(),
    #                  ckpt_path=config['resume_checkpoint'] if config['resume_checkpoint'] != "" else None)
    #



if __name__ == '__main__':
    main()
