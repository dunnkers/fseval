from typing import Any, Dict, Optional

from fseval.types import Callback
from omegaconf import OmegaConf
from yaml import dump


class CallbackCollection(Callback):
    def __init__(self, callbacks: Dict = {}):
        super(CallbackCollection, self).__init__()

        callback_names = []
        for callback_name, callback in callbacks.items():
            callback_names.append(callback_name)
            setattr(self, callback_name, callback)

        self.callback_names = callback_names

    @property
    def _iterator(self):
        return [getattr(self, callback_name) for callback_name in self.callback_names]

    def set_config(self, config: Dict):
        self.config = config

        for callback in self._iterator:
            callback.set_config(config)

    def on_begin(self):
        for callback in self._iterator:
            callback.on_begin()

    def on_config_update(self, config: Dict):
        for callback in self._iterator:
            callback.on_config_update(config)

    def on_log(self, msg: Any, *args: Any):
        for callback in self._iterator:
            callback.on_log(msg, *args)

    def on_metrics(self, metrics):
        for callback in self._iterator:
            callback.on_metrics(metrics)

    def on_summary(self, summary):
        for callback in self._iterator:
            callback.on_summary(summary)

    def on_end(self, exit_code: Optional[int] = None):
        for callback in reversed(self._iterator):
            callback.on_end(exit_code)
