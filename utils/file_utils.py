import os
import pandas as pd
import yaml


class FileUtils:
    """
    ファイル操作に関する処理を提供するクラス
    """

    @staticmethod
    def load_yaml_as_dict(path: str) -> dict:
        """
        .ymlファイルをロードし、辞書型に変換して返す関数
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_dict_to_yaml(dictionary: dict, path: str):
        """
        辞書型を.ymlとして保存する関数
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(dictionary, f, allow_unicode=True)

    @staticmethod
    def save_dict_to_csv(dictionary: dict, path: str):
        """
        単一の辞書を1行としてCSVに追記保存する関数（ヘッダー付き・自動判断）
        """
        df = pd.DataFrame([dictionary])
        write_header = not os.path.exists(path)
        df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8")
