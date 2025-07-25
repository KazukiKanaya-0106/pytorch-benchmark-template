class DisplayUtils:
    @staticmethod
    def print_metrics(metrics: dict, title: str = "Metrics") -> None:
        print(f"\n{title}")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:<35}: {value:>10.4f}")
            else:
                print(f"{key:<35}: {value:>10}")
        print("=" * 50)
