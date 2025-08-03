from utils import DataStructureUtils, FileUtils


class GridSearchParameters:
    """
    グリッドサーチのパラメータを辞書型として生成するクラス
    """

    def __init__(self, grid_search_params_path: str) -> None:
        raw_params_dict: dict = FileUtils.load_yaml_as_dict(grid_search_params_path)
        params_dict: dict = DataStructureUtils.convert_to_builtin_types(raw_params_dict)
        self._params_dict = params_dict

    @property
    def parameters(self) -> dict:
        return self._params_dict

    @property
    def flattened_parameters(self) -> dict:
        return DataStructureUtils.flatten_dict(self._params_dict)
