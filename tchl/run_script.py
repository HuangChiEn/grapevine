
# * torchlightning, 2.2.3 version
# ? plz strictly chk torch competible version grid!
# ? src: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
import lightning as L 
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

# * our dependencies for trainer_callback, model, dataset
from trainer_callbk import add_lr_monitor
from model import build_model, Model
from dataset import build_data_loader, build_data_module

def main(cfger):
    # * 0. seed setup
    seed_everything(cfger.seed, workers=True)

    # * 1. get model & dataset
    model = build_model(cfger.model_params)

    # ? optional for building data_loader
    #tra_ld, val_ld, tst_ld = build_data_loader(cfger.data_params)
    data_module = build_data_module(cfger.dataset_params)
    
    # * 2. setup trainer utils
    # ? add callback via list (shallow copy allow None return)
    callbk_lst = []
    add_lr_monitor(callbk_lst, logging_interval='step')

    # ? wandb logger plugin by lightning, more info 
    # ? plz ref:https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#wandb
    logger = WandbLogger(**cfger.logger['setup']) if cfger.logger['log_it'] else None 
    logger and logger.log_hyperparams( vars(cfger) )
    # logger.watch(model, log="all") # logging gradient, network, ..., etc.

    # * 3. setup tchl trainer
    # ? more setup about pl.Trainer, 
    # ? plz ref:https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#trainer
    trainer = L.Trainer(**cfger.trainer, logger=logger, callbacks=callbk_lst)
    
    # * 4. simple subroutine dispatcher
    # ? dict-mapping based dispatcher is also fine ~
    if cfger.exe_stage == 'train':
        # trainer.fit(model, train_dataloaders=tra_ld, val_dataloaders=val_ld)
        trainer.fit(
            model=model, 
            datamodule=data_module, 
            ckpt_path=cfger.train_params.resume_path
        )

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

    # * about checkpoint saving mechanism (it's fully automatic in torchlightning)
    # ? plz ref:https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#saving-and-loading-checkpoints-basic


# * prepare config & entry point main
if __name__ == "__main__":
    from easy_configer.Configer import Configer
    
    # ? allow CLI to runtime decide config path, append `cfg_path=/my/cfgs/sample_config.ini@str`
    cfger = Configer(cmd_args=True)
    cfg_path = cfger.cfg_path if 'cfg_path' in cfger else './sample_config.ini'
    
    cfger.cfg_from_ini(cfg_path)
    main(cfger)