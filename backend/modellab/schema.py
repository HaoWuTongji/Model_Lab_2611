"""The sole symbolic contract; presentation text never executes."""

from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr

Id = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]
Member = Union[StrictInt, StrictStr]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    image_id: str
    order: int = 0
    bbox: tuple[float, float, float, float] | None = None


class Index(Strict):
    kind: Literal["index"]
    name: Id


class LiteralIndex(Strict):
    kind: Literal["literal"]
    value: Member


IndexRef = Annotated[Union[Index, LiteralIndex], Field(discriminator="kind")]


class Number(Strict):
    kind: Literal["number"]
    value: FiniteFloat


class Param(Strict):
    kind: Literal["param"]
    id: Id
    indices: list[IndexRef] = []


class Var(Strict):
    kind: Literal["var"]
    id: Id
    indices: list[IndexRef] = []


class Add(Strict):
    kind: Literal["add"]
    args: list["Expr"]


class Neg(Strict):
    kind: Literal["neg"]
    arg: "Expr"


class Mul(Strict):
    kind: Literal["mul"]
    left: "Expr"
    right: "Expr"


class Div(Strict):
    kind: Literal["div"]
    numerator: "Expr"
    denominator: "Expr"


class Sum(Strict):
    kind: Literal["sum"]
    index: Id
    set: Id
    set_index: IndexRef | None = None
    body: "Expr"


Expr = Annotated[
    Union[Number, Param, Var, Add, Neg, Mul, Div, Sum], Field(discriminator="kind")
]


class IndexedSetEntry(Strict):
    index: Member
    members: list[Member]


class SetDecl(Strict):
    id: Id
    members: list[Member] = []
    index_set: Id | None = None
    indexed_members: list[IndexedSetEntry] = []
    description: str = ""


class Entry(Strict):
    indices: list[Member]
    value: FiniteFloat


class Parameter(Strict):
    id: Id
    symbol: str = ""
    description: str = ""
    sets: list[Id] = []
    value: FiniteFloat | None = None
    entries: list[Entry] = []
    data_key: Id | None = None
    default: FiniteFloat | None = None
    source_refs: list[Source] = []


class Variable(Strict):
    id: Id
    symbol: str = ""
    description: str = ""
    sets: list[Id] = []
    domain: Literal["continuous", "integer", "binary"]
    lower: FiniteFloat | None  # Required: absent is unresolved, explicit null is free.
    upper: FiniteFloat | None
    range_source: str = "manual"
    source_refs: list[Source] = []


class Quantifier(Strict):
    index: Id
    set: Id
    set_index: IndexRef | None = None


class Constraint(Strict):
    id: Id
    description: str = ""
    for_each: list[Quantifier] = []
    where: str = ""
    lhs: Expr
    sense: Literal["le", "eq", "ge"]
    rhs: Expr
    source_refs: list[Source] = []
    transcript_latex: str = ""


class Objective(Strict):
    id: Id = "obj"
    description: str = ""
    sense: Literal["min", "max"]
    expr: Expr
    source_refs: list[Source] = []
    transcript_latex: str = ""


class Unresolved(Strict):
    id: str
    source_id: str = ""
    question: str
    resolved: bool = False
    resolution: str = ""


class ModelDoc(Strict):
    schema_version: Literal["1.0"] = "1.0"
    sets: list[SetDecl] = []
    parameters: list[Parameter] = []
    variables: list[Variable]
    objective: Objective
    constraints: list[Constraint] = []
    unresolved_items: list[Unresolved] = []
    transcript: str = ""


for cls in (Add, Neg, Mul, Div, Sum, Constraint, Objective, ModelDoc):
    cls.model_rebuild()
