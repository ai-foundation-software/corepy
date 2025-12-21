from typing import Any, List, Callable
from corepy.data import Table

class Pipeline:
    """
    A linear execution pipeline for data transformations.
    """
    def __init__(self, steps: List[Callable[[Table], Table]] = None):
        self.steps = steps or []

    def add_step(self, step: Callable[[Table], Table]) -> None:
        """
        Adds a transformation step to the pipeline.
        """
        self.steps.append(step)

    def run(self, data: Table) -> Table:
        """
        Executes the pipeline on the given data.
        """
        result = data
        for step in self.steps:
            result = step(result)
        return result

    def __repr__(self) -> str:
        return f"Pipeline(steps={len(self.steps)})"
