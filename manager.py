import logging
import math
import statistics
import time

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# Import classes from algorithms.py
from algorithms import PrintOptimizer, RodCutter


########################################
# A small helper to measure time
########################################
def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


########################################
# Rough complexity estimation (like before)
########################################
def estimate_complexity(sizes, times):
    if len(sizes) < 2:
        return {"best_label": "Not enough data"}
    import statistics

    candidates = {
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * math.log2(n) if n > 1 else n,
        "O(n^2)": lambda n: n**2,
        "O(log n)": lambda n: math.log2(n) if n > 1 else 1,
    }
    valid_data = [(n, t) for (n, t) in zip(sizes, times) if n > 0 and t > 0]
    if len(valid_data) < 2:
        return {"best_label": "Not enough data"}

    results = {}
    for label, f in candidates.items():
        ratio = []
        for n, t in valid_data:
            val = f(n)
            if val == 0:
                continue
            ratio.append(t / val)
        if len(ratio) < 2:
            results[label] = (float("inf"), float("inf"))
            continue
        mean_v = statistics.mean(ratio)
        std_v = statistics.pstdev(ratio)
        results[label] = (mean_v, std_v)

    best_label = None
    best_std = float("inf")
    for label, (m, s) in results.items():
        if s < best_std:
            best_std = s
            best_label = label
    return {"best_label": best_label}


########################################
# Manager for Task 1 (3D-printer)
########################################
class PrintOptimizerManager:
    def __init__(self):
        self.optimizer = PrintOptimizer()

    def run_tests(self):
        """
        We'll replicate exactly the 3 test cases from the assignment:
         1) All same priority,
         2) Different priorities,
         3) Exceed constraints.
        We'll store results in a DataFrame to compare with expected.
        """
        constraints = {"max_volume": 300, "max_items": 2}

        # Test1: All same priority
        test1_jobs = [
            {"id": "M1", "volume": 100, "priority": 1, "print_time": 120},
            {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},
            {"id": "M3", "volume": 120, "priority": 1, "print_time": 150},
        ]

        # Test2: Different priorities
        test2_jobs = [
            {"id": "M1", "volume": 100, "priority": 2, "print_time": 120},  # Lab
            {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},  # Diploma
            {"id": "M3", "volume": 120, "priority": 3, "print_time": 150},  # Personal
        ]

        # Test3: Exceed constraints
        test3_jobs = [
            {"id": "M1", "volume": 250, "priority": 1, "print_time": 180},
            {"id": "M2", "volume": 200, "priority": 1, "print_time": 150},
            {"id": "M3", "volume": 180, "priority": 2, "print_time": 120},
        ]

        test_cases = [
            ("Test1_same_priority", test1_jobs),
            ("Test2_different_priority", test2_jobs),
            ("Test3_exceed_constraints", test3_jobs),
        ]

        records = []
        for test_name, jobs in test_cases:
            (res, elapsed) = measure_time(
                self.optimizer.optimize_printing, jobs, constraints
            )
            records.append(
                {
                    "TestName": test_name,
                    "Algorithm": "PrintOptimizer",
                    "PrintOrder": res["print_order"],
                    "TotalTime": res["total_time"],
                    "Time(s)": elapsed,
                }
            )

        df = pd.DataFrame(records)
        return df

    def plot_results(self, df):
        """
        If we had multiple sizes, we could plot them. But here we only have 3 test cases.
        We might skip real plotting or do a bar chart for 'Time(s)'.
        """
        # Example: bar chart by TestName
        plt.figure(figsize=(7, 4))
        plt.bar(df["TestName"], df["Time(s)"], color="skyblue")
        plt.title("PrintOptimizer: Execution Time by Test")
        plt.xlabel("Test Name")
        plt.ylabel("Time (s)")
        plt.tight_layout()
        plt.show()


########################################
# Manager for Task 2 (Rod Cutting)
########################################
class RodCutterManager:
    def __init__(self):
        self.cutter = RodCutter()

    def run_tests(self):
        """
        We replicate the 3 test cases from the assignment:
         1) length=5, prices=[2,5,7,8,10],
         2) length=3, prices=[1,3,8],
         3) length=4, prices=[3,5,6,7].
        We'll run both memo & table for each test, storing results in DF.
        """
        test_cases = [
            ("BaseCase", 5, [2, 5, 7, 8, 10]),
            ("NoCutNeeded", 3, [1, 3, 8]),
            ("EvenCuts", 4, [3, 5, 6, 7]),
        ]

        records = []
        for test_name, length, prices in test_cases:
            # memo
            (memo_res, memo_time) = measure_time(
                self.cutter.rod_cutting_memo, length, prices
            )
            records.append(
                {
                    "TestName": test_name,
                    "Method": "Memo",
                    "Length": length,
                    "Prices": prices,
                    "MaxProfit": memo_res["max_profit"],
                    "Cuts": memo_res["cuts"],
                    "NumberOfCuts": memo_res["number_of_cuts"],
                    "Time(s)": memo_time,
                }
            )

            # table
            (tab_res, tab_time) = measure_time(
                self.cutter.rod_cutting_table, length, prices
            )
            records.append(
                {
                    "TestName": test_name,
                    "Method": "Table",
                    "Length": length,
                    "Prices": prices,
                    "MaxProfit": tab_res["max_profit"],
                    "Cuts": tab_res["cuts"],
                    "NumberOfCuts": tab_res["number_of_cuts"],
                    "Time(s)": tab_time,
                }
            )
        df = pd.DataFrame(records)
        return df

    def plot_results(self, df):
        """
        We'll do a bar chart comparing Memo vs Table time for each test.
        """
        # grouping by TestName
        grouped = df.groupby("TestName")

        # or we can just do a single bar for each row
        plt.figure(figsize=(7, 4))
        x_ticks = range(len(df))
        plt.bar(x_ticks, df["Time(s)"], color="orange")
        plt.xticks(x_ticks, df["TestName"] + "_" + df["Method"], rotation=45)
        plt.title("Rod Cutting: Execution Time by Test")
        plt.xlabel("Test + Method")
        plt.ylabel("Time(s)")
        plt.tight_layout()
        plt.show()


########################################
# MainManager orchestrates both
########################################
class MainManager:
    def __init__(self):
        self.print_mgr = PrintOptimizerManager()
        self.rod_mgr = RodCutterManager()

    def finalize_analysis(self, df, algo_col, time_col):
        """
        If we had multiple sizes, we could estimate complexity.
        But here we only have 3 tests for printing,
        and also 3 tests for rod cutting => not enough data to do a nice complexity analysis.
        We'll attempt anyway to demonstrate.
        """
        # We'll group by the "Algorithm" or "Method" column
        methods = df[algo_col].unique()
        for m in methods:
            sub = df[df[algo_col] == m]
            # If sub has <2 points, skip
            if len(sub) < 2:
                logger.info(f"Skipping complexity estimate for {m}: not enough data.")
                continue
            # We do "Test index" as size
            sizes_list = list(range(1, len(sub) + 1))
            times_list = sub[time_col].tolist()
            from math import log2
            import statistics

            # naive usage
            info = estimate_complexity(sizes_list, times_list)
            best_label = info.get("best_label", "N/A")
            logger.info(
                Fore.YELLOW
                + f"Empirical complexity for '{m}': {best_label}"
                + Style.RESET_ALL
            )

    def main(self):
        logger.info(
            Fore.GREEN + "--- Testing 3D Printer Optimizer ---" + Style.RESET_ALL
        )
        df_print = self.print_mgr.run_tests()
        logger.info(Fore.CYAN + "\n[3D Printer] Results:\n" + Style.RESET_ALL)
        print(
            tabulate(
                df_print,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False,
            )
        )
        self.print_mgr.plot_results(df_print)
        self.finalize_analysis(df_print, "Algorithm", "Time(s)")

        logger.info(Fore.GREEN + "--- Testing Rod Cutting ---" + Style.RESET_ALL)
        df_rod = self.rod_mgr.run_tests()
        logger.info(Fore.CYAN + "\n[Rod Cutting] Results:\n" + Style.RESET_ALL)
        print(
            tabulate(
                df_rod,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False,
            )
        )
        self.rod_mgr.plot_results(df_rod)
        self.finalize_analysis(df_rod, "Method", "Time(s)")

        logger.info(Fore.MAGENTA + "All tests finished." + Style.RESET_ALL)


if __name__ == "__main__":
    app = MainManager()
    app.main()
