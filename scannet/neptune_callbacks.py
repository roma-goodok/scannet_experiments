"A `Callback` that saves tracked metrics into a Neptune"
#Contribution from devforfu: https://nbviewer.jupyter.org/gist/devforfu/ea0b3fcfe194dad323c3762492b05cae

from fastprogress.fastprogress import format_time

from fastai.torch_core import *
from fastai.basic_data import DataBunch
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback

import neptune

from time import time

__all__ = ['NeptuneMonitor']

# Fixes
# - flush  on_epoch_end
# - learn.add_time

class NeptuneMonitor(LearnerCallback):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."
    def __init__(self, learn:Learner, neptune_experiment): 
        super().__init__(learn)
        self.neptune_experiment = neptune_experiment

    def on_train_begin(self, **kwargs: Any) -> None:
        pass
        
    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        
        for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics):
            #if name not in ['epoch']:
            self.neptune_experiment.send_metric(name, epoch, stat)
    
    def on_train_end(self, **kwargs: Any) -> None:  
        pass
