from typing import List


class HandType:

    def __init__(self, category: str, primary_value: int, kickers: List[int]):
        self.category = category # eg. "three_of_a_kind"
        self.primary_value = primary_value # The main value defining the hand eg. one pair of 5
        kickers.sort(reverse=True)
        self.kickers = kickers # A list of kicker values for tie-breaking, sorted in descending order
