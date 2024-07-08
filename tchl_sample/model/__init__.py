
# * we provide several kind of torchlightning model..

# ? How to intergreate sub-modules (nn.Module) in a single torchlightning model 
from .LiteAutoEncoder import AutoEncoder as Model
# ? How to manuelly optimizer (for GAN or other customized optimization behavior)
from .LiteGAN import GAN

# * this is how we pass args to init the model
def build_model(*args, **kwagrs):
    # ? This script just show how we build the system,
    # ? so, AutoEncoder have their own kwargs
    return Model()