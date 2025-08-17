from tabulate import tabulate


class DisplayUtils:
    @staticmethod
    def print_metrics(metrics: dict, title: str = "Metrics") -> None:
        print(f"\n[{title}]")
        table = []
        for key, value in metrics.items():
            if isinstance(value, float):
                table.append([key, f"{value:.4f}"])
            else:
                table.append([key, str(value)])
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="github"))
