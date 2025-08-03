import datetime
from utils import DataStructureUtils, FileUtils


class Config:
    """
    YAMLベースの設定を管理・マージ・拡張するクラス
    """

    def __init__(self, base_path: str, override_path: str, key: str) -> None:
        base_dict: dict = FileUtils.load_yaml_as_dict(base_path)
        override_dict: dict = FileUtils.load_yaml_as_dict(override_path)

        raw_merged_dict: dict = DataStructureUtils.deep_merge_dict(base_dict=base_dict, override_dict=override_dict)
        merged_dict: dict = DataStructureUtils.convert_to_builtin_types(raw_merged_dict)

        # タイムスタンプ追加（JST）
        JST = datetime.timezone(datetime.timedelta(hours=9))
        timestamp = datetime.datetime.now(JST).strftime("%Y-%m-%d_%H-%M-%S")
        merged_dict["meta"].update({"timestamp": timestamp, "key": f"{timestamp}_{key}" if key else timestamp})

        self._config: dict = merged_dict

    @property
    def config(self) -> dict:
        return self._config
