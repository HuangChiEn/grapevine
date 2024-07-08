from lightning.pytorch.callbacks import (
    LearningRateMonitor, 
    ModelCheckpoint, 
    ModelSummary, 
    RichProgressBar,
    EarlyStopping
)
# * Lightning support built-in callback :
# * https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#built-in-callbacks

# ? src: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#learningratemonitor
def add_lr_monitor(callbk_lst: list, **kwargs) -> None :
    callbk_lst.append( LearningRateMonitor(**kwargs) )

# ? src: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
def add_model_ckpt(callbk_lst: list, **kwargs) -> None :
    callbk_lst.append( ModelCheckpoint(**kwargs) )

# ? src: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelSummary.html#modelsummary
def add_model_summary(callbk_lst: list, **kwargs) -> None:
    callbk_lst.append( ModelSummary(**kwargs) )

# ? src: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichProgressBar.html#richprogressbar
def add_rich_probar(callbk_lst: list, **kwargs) -> None:
    callbk_lst.append( RichProgressBar(**kwargs) )

# ? src: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#earlystopping
def add_early_stop(callbk_lst: list, **kwargs) -> None:
    callbk_lst.append( EarlyStopping(**kwargs) )
