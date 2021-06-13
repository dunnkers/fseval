from typing import Dict, Optional

from omegaconf import DictConfig

from fseval.types import Callback


class CallbackCollection(Callback):
    def __init__(self, **callbacks):
        super(CallbackCollection, self).__init__()

        callback_names = []
        for callback_name, callback in callbacks.items():
            callback_names.append(callback_name)
            setattr(self, callback_name, callback)

        self.callback_names = callback_names

    @property
    def _iterator(self):
        return [getattr(self, callback_name) for callback_name in self.callback_names]

    def on_begin(self, config: DictConfig):
        for callback in self._iterator:
            callback.on_begin(config)

    def on_config_update(self, config: Dict):
        for callback in self._iterator:
            callback.on_config_update(config)

    def on_metrics(self, metrics):
        for callback in self._iterator:
            callback.on_metrics(metrics)

    def on_summary(self, summary):
        for callback in self._iterator:
            callback.on_summary(summary)

    def on_end(self, exit_code: Optional[int] = None):
        for callback in reversed(self._iterator):
            callback.on_end(exit_code)
