from abc import ABC
from typing import Any, Dict, List


class IOHandler(ABC):
    def __init__(self):
        self.pipeline_config = None

    def set_pipeline_config(self, pipeline_config: Dict):
        self.pipeline_config = pipeline_config

    def on_file_restore(self, filename) -> Any:
        ...

    def on_file_save(self, filename, content):
        ...

class WandbIOHandler:
    def on_file_restore(self, filename) -> Any:
        raise NotImplementedError

    def on_file_save(self, filename, content):
        filepath = os.path.join(wandb.run.dir, filename)
        f = open(filepath, "w")
        f.write(content)
        f.close()

        wandb.save(filename, base_path="/")

