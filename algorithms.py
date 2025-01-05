import logging
import math
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class PrintJob:
    id: str
    volume: float
    priority: int
    print_time: int


@dataclass
class PrinterConstraints:
    max_volume: float
    max_items: int


class PrintOptimizer:
    """
    Implements a greedy approach to optimize 3D printer queue:
      - Sort jobs by priority (1=highest).
      - Greedily group them under max_volume & max_items constraints.
      - The time for a group = max(print_time) among its jobs.
      - Sum times of all groups -> total_time.
    """

    def optimize_printing(self, print_jobs: List[Dict], constraints: Dict) -> Dict:
        # Convert to dataclass
        jobs = [PrintJob(**job) for job in print_jobs]
        printer = PrinterConstraints(**constraints)

        # Sort by priority ascending (1 - highest)
        jobs.sort(key=lambda x: x.priority)

        total_time = 0
        print_order = []
        current_batch = []
        current_volume = 0

        def finalize_batch(batch):
            if not batch:
                return 0
            batch_time = max(job.print_time for job in batch)
            return batch_time

        for job in jobs:
            # If adding this job breaks constraints => finalize previous batch
            if current_batch and (
                current_volume + job.volume > printer.max_volume
                or len(current_batch) >= printer.max_items
            ):
                # finalize
                b_time = finalize_batch(current_batch)
                total_time += b_time
                print_order.extend([j.id for j in current_batch])
                # new batch
                current_batch = []
                current_volume = 0

            current_batch.append(job)
            current_volume += job.volume

        # finalize the last batch
        if current_batch:
            b_time = finalize_batch(current_batch)
            total_time += b_time
            print_order.extend([j.id for j in current_batch])

        return {"print_order": print_order, "total_time": total_time}


class RodCutter:
    """
    Provides two methods to solve the Rod Cutting problem:
    1) Memoization (rod_cutting_memo)
    2) Tabulation (rod_cutting_table)
    Each returns dict:
      {
        "max_profit": <int>,
        "cuts": <List[int]>,
        "number_of_cuts": <int>
      }
    """

    def rod_cutting_memo(self, length: int, prices: List[int]) -> Dict:
        memo = {}

        def recurse(n):
            if n == 0:
                return (0, [])
            if n in memo:
                return memo[n]
            best_profit = 0
            best_cuts = []
            for first_cut in range(1, n + 1):
                current_price = prices[first_cut - 1]
                (rem_profit, rem_cuts) = recurse(n - first_cut)
                total_price = current_price + rem_profit
                if total_price > best_profit:
                    best_profit = total_price
                    best_cuts = [first_cut] + rem_cuts
            memo[n] = (best_profit, best_cuts)
            return memo[n]

        (max_profit, cuts) = recurse(length)
        return {
            "max_profit": max_profit,
            "cuts": cuts,
            "number_of_cuts": len(cuts) - 1 if cuts else 0,
        }

    def rod_cutting_table(self, length: int, prices: List[int]) -> Dict:
        dp = [(0, []) for _ in range(length + 1)]
        for n in range(1, length + 1):
            best_profit = 0
            best_cuts = []
            for first_cut in range(1, n + 1):
                current_price = prices[first_cut - 1]
                (rem_profit, rem_cuts) = dp[n - first_cut]
                total_price = current_price + rem_profit
                if total_price > best_profit:
                    best_profit = total_price
                    best_cuts = [first_cut] + rem_cuts
            dp[n] = (best_profit, best_cuts)
        (max_profit, cuts) = dp[length]
        return {
            "max_profit": max_profit,
            "cuts": cuts,
            "number_of_cuts": len(cuts) - 1 if cuts else 0,
        }
