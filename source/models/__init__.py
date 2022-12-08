from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer


def model_factory(config: DictConfig, repeat_index):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config, repeat_index).cuda()
