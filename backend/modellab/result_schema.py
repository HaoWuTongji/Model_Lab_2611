from typing import Literal
from pydantic import FiniteFloat, StrictBool, model_validator
from .schema import Strict


class SolverResult(Strict):
    termination: Literal[
        "optimal",
        "infeasible",
        "unbounded",
        "infeasible_or_unbounded",
        "time_limit",
        "iteration_limit",
        "cancelled",
        "numerical_error",
        "input_error",
        "execution_error",
        "unknown",
    ]
    has_incumbent: StrictBool
    proven_optimal: StrictBool
    objective_value: FiniteFloat | None
    best_bound: FiniteFloat | None = None
    relative_gap: FiniteFloat | None = None
    variables: dict[str, FiniteFloat] | None
    wall_time: FiniteFloat = 0
    solver_time: FiniteFloat = 0
    raw_status: str
    solver_version: str
    conflict_sources: list[str] | None = None

    @model_validator(mode="after")
    def consistency(self):
        if self.has_incumbent and (
            self.variables is None or self.objective_value is None
        ):
            raise ValueError("incumbent requires values")
        if not self.has_incumbent and (
            self.variables is not None or self.objective_value is not None
        ):
            raise ValueError("no incumbent must have null values")
        if self.proven_optimal and (
            self.termination != "optimal" or not self.has_incumbent
        ):
            raise ValueError("inconsistent optimality")
        return self
