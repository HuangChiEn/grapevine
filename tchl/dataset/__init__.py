
# ? How to modularize the dataset/dataloader in a single class 
from .get_datamodule import MNISTDataModule as DataModule
# ? How to manuelly make torch.dataset / loader
from .get_dataset import Dummy_tch_ds

# ? recommended way to install torch with its ecosystem
# ? src: https://pytorch.org/get-started/previous-versions/ 
import torch.utils.data as tch_ds 
from easy_configer.Configer import cfgtyp
from typing import List

import lightning as L 

def build_data_loader(cfger: cfgtyp.AttributeDict) -> List[tch_ds.DataLoader] :
    dummy_ds = Dummy_tch_ds()
    return dummy_ds.get_all_dataloader()

def build_data_module(cfger: cfgtyp.AttributeDict) -> L.LightningDataModule :
    return DataModule()