from fseval.config import PipelineConfig


class ShouldTestGroupItem:
    @staticmethod
    def should_test(cfg: PipelineConfig, group_name: str) -> bool:
        return True
