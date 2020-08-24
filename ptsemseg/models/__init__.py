import copy
import torchvision.models as models

from ptsemseg.models.hardnet import hardnet
from ptsemseg.models.icnet import icnet
from ptsemseg.models.bisenet import BiSeNetV2


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "hardnet": hardnet,
            "icnet": icnet,
            "bisenetv2": BiSeNetV2,

        }[name]
    except:
        raise ("Model {} not available".format(name))
