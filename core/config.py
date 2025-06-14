import datetime
from utils import DataStructureUtils, FileUtils


class Config:
    """
    configの情報を管理するコンポーネント
    """

    def __init__(self, base_path: str, override_path: str, key: str) -> None:
        base_dict: dict = FileUtils.load_yaml_as_dict(base_path)
        override_dict: dict = DataStructureUtils.load_yaml_as_dict(override_path)
        merged_dict: dict = DataStructureUtils.deep_merge_dict(
            base_dict=base_dict, override_dict=override_dict
        )
        converted_dict: dict = DataStructureUtils.convert_to_builtin_types(merged_dict)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        converted_dict["meta"] = {
            "timestamp": timestamp,
            "key": f"{timestamp}_{key}" if key else f"{timestamp}",
        }

        self.config: dict = converted_dict

    def get(self) -> dict:
        return self.config
