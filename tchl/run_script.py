from easy_configer.Configer import cfgtyp

# * torchlightning, 2.2.3 version
# ? plz strictly chk torch competible version grid!
# ? src: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
import lightning as L 
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

# ? recommended way to install torch
# ? src: https://pytorch.org/get-started/previous-versions/ 
import torch.utils.data as tch_ds 

from trainer_callbk import add_lr_monitor

# ? this script is standalone, we Mock (simulate) all objects
from unittest.mock import Mock
Model = Mock(spec=L.LightningModule)
DataModule = Mock(spec=L.LightningDataModule)
DataLoader = Mock(spec=tch_ds.DataLoader)

def build_model(cfger: cfgtyp.AttributeDict) -> L.LightningModule :
    return Model(**cfger)

def build_data_module(cfger: cfgtyp.AttributeDict) -> L.LightningDataModule :
    return DataModule(**cfger)

def build_data_loader(cfger: cfgtyp.AttributeDict) -> tch_ds.DataLoader :
    return DataLoader(**cfger)

def main(cfger):
    # * 0. seed setup
    seed_everything(cfger.seed, workers=True)

    # * 1. get model & dataset
    model = build_model(cfger.model_params)

    # ? optional for building data_loader
    #tra_ld, val_ld, tst_ld = build_data_loader(cfger.data_params)
    data_module = build_data_module(cfger.data_params)
    
    # * 2. setup trainer utils
    # ? add callback via list (shallow copy allow None return)
    callbk_lst = []
    add_lr_monitor(callbk_lst, logging_interval='step')

    # ? wandb logger plugin by lightning 
    logger = WandbLogger(**cfger.logger['setup']) if cfger.logger['log_it'] else None 
    logger.log_hyperparams( vars(cfger) )
    # logger.watch(model, log="all")

    # * 3. setup tchl trainer
    trainer = L.Trainer(**cfger.trainer, logger=logger, callbacks=callbk_lst)
    
    # * 4. simple subroutine dispatcher
    # ? dict-mapping based dispatcher is also fine ~
    if cfger.exe_stage == 'train':
        trainer.fit(model=model, datamodule=data_module, ckpt_path=cfger.train_params.resume_path)

    elif cfger.exe_stage == 'finetune':
        model = Model.load_from_checkpoint(
            checkpoint_path=cfger.finetune_params['PATH'], 
            **cfger.finetune_params
        )
        trainer.fit(model=model, datamodule=data_module)

    else:
        model = Model.load_from_checkpoint(
            checkpoint_path=cfger.test_params['PATH'], 
            **cfger.test_params
        )
        trainer.test(model=model, datamodule=data_module)


# * prepare config & entry point main
if __name__ == "__main__":
    import os
    from easy_configer.Configer import Configer
    cfg_path = os.environ['CONFIGER_PATH'] if os.environ['CONFIGER_PATH'] \
                                           else './sample_config.ini'
    
    # ? allow CLI to feed args 
    cfger = Configer(cmd_args=True)

    cfger.cfg_from_ini(cfg_path)
    main(cfger)