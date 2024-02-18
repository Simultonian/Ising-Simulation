class Counter:
    def __init__(self):
        self.count = {}

    def add(self, count: dict[str, int]):
        for ky, val in count.items():
            self.count[ky] = self.count.get(ky, 0) + val

    def weighted_add(self, weight: int, count: dict[str, int]):
        for ky, val in count.items():
            self.count[ky] = self.count.get(ky, 0) + (weight * val)

    def times(self, val: int) -> dict[str, int]:
        return {ky: vl * val for ky, vl in self.count.items()}
