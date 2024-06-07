
import torch
from torch import optim, nn
import lightning as L

# define any number of nn.Modules (or use your current ones)
Encoder = lambda in_dim, hid_dim, out_dim : nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim))
Decoder = lambda in_dim, hid_dim, out_dim : nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim))

# define the LightningModule
class AutoEncoder(L.LightningModule):

    def __init__(
            self, 
            enc_kwargs={'in_dim':28*28, 'hid_dim':64, 'out_dim':3}, 
            dec_kwargs={'in_dim':3, 'hid_dim':64, 'out_dim':28*28}
        ):
        # * must call constructor at fist line!
        super().__init__()
        # * 'automatically' save all parameters passing to __init__(.),
        # ? also, don't forget to only saving hyperparams for init submodules!!
        all_kwargs = {**enc_kwargs, **dec_kwargs}
        self.save_hyperparameters(all_kwargs)

        # * Recommended way : https://github.com/Lightning-AI/pytorch-lightning/discussions/13615#discussioncomment-3150654
        # init torch.nn Module in lightningmodule self.__init__(.)
        self.encoder = Encoder(**enc_kwargs)
        self.decoder = Decoder(**dec_kwargs)

    # * a blackbox procedure for lower-level operation in model
    def forward(self, x):
        # ? for ex, we consider a blackbox to compress & decompress latent in VAE,
        # ?  user don't need to aware in training_step
        latent = self.encoder(x)
        return self.decoder(latent)

    # * training_step defines the train loop.
    def training_step(self, batch, batch_idx):
        # unroll batch content..
        x, y = batch
        x = x.view(x.size(0), -1)

        # ? call self.forward(.)
        x_hat = self(x)
        
        # loss for supervised modules
        loss = nn.functional.mse_loss(x_hat, x)
        
        # * Logging to TensorBoard (if installed) by default
        # ? prog_bar to show in console, more setup about self.log, self.log_dict
        # ? plz ref:https://lightning.ai/docs/pytorch/stable/extensions/logging.html#automatic-logging
        self.log("train_loss", loss, prog_bar=True)

        # ? log non-metric infor, plz ref the above link,
        # ? for wandb : https://docs.wandb.ai/guides/integrations/lightning#log-images-text-and-more
        # tensorboard = self.logger.experiment
        # tensorboard.add_image()

        # get the dynamic information about pl.Trainer by the object `self.trainer`
        # ? i.e lr, get the first optimizer's lr wrt. network parameters in group 0
        # ? self.trainer.optimizers[0].param_groups[0]["lr"]
        
        # * it's important, you should return the sum of losses at the end of train_step!!
        return loss
    
    # * for validation_step, test_step
    # ? just self.log(.) the metric, don't need to return loss in following steps!!
    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ...  

    def configure_optimizers(self):
        # this show how to setup optimizer & scheduler 
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        # ? for multiple optimizer and scheduler, just append at the lists
        return [optimizer], [sch]
