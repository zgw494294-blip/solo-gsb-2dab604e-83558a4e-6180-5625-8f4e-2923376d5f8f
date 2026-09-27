"""带单位线性方程组的精确求解。

POST /solve-quantities 的核心计算：

- 1～8 个互异变量（name、unit）与 1～16 条方程；每条方程形如
  ``Σ coefficient·unit·var = value·unit``，``unit="1"`` 表示无量纲；
- 逐项核对量纲：``dims(term.unit) + dims(var.unit) == dims(rhs.unit)``；
- 系数与右侧值按单位比例换算到基本单位（m/kg/s/K），变量值以基本单位
  表示，组成有理系数增广矩阵 [A | b]，用 ``Fraction`` 做 Gauss-Jordan
  精确消元；
- 仅唯一解被接受：x_i 除以变量声明单位的比例换回声明单位；矛盾（无解）
  与欠定分别抛出不同的错误类型。

``°C`` / ``°F`` 是偏移温标，没有唯一的乘性比例，所有单位一律拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .converter import (
    InvalidValueError,
    OffsetScaleInCompoundError,
    Quantity,
    UnitSyntaxError,
    dim_description,
    parse_expression,
    parse_value,
)


# ---------------------------------------------------------------------------
# 输入结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableInput:
    """一个待求解变量：name 互异（请求层校验），unit 为单位表达式。"""
    name: str
    unit: str


@dataclass(frozen=True)
class TermInput:
    """方程左侧的一项：var 引用已声明变量，coefficient 为原始字符串。"""
    var: str
    coefficient: str
    unit: str


@dataclass(frozen=True)
class RhsInput:
    """方程右侧：value 为原始字符串。"""
    value: str
    unit: str


@dataclass(frozen=True)
class EquationInput:
    terms: tuple[TermInput, ...]
    rhs: RhsInput


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class SolveError(Exception):
    """消元失败的基类；error 为对外的错误类型标识。"""

    error = "solve_error"

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


class NoSolutionError(SolveError):
    error = "no_solution"


class UnderdeterminedError(SolveError):
    error = "underdetermined"


@dataclass(frozen=True)
class UnitError(Exception):
    """单位表达式无法用于方程求解。

    kind 为 "syntax_error" / "offset_temperature_not_allowed"；坐标为
    variable（变量下标），或 equation + term（项下标；term 为 None
    表示该式右侧）。
    """
    message: str
    kind: str
    position: int | None
    variable: int | None = None
    equation: int | None = None
    term: int | None = None


@dataclass(frozen=True)
class InvalidNumberError(Exception):
    """coefficient / rhs.value 不是有限十进制字符串；term 为 None 表示右侧。"""
    message: str
    equation: int
    term: int | None


@dataclass(frozen=True)
class TermDimensionError(Exception):
    """某项的 系数单位×变量单位 与该式右侧量纲不一致。"""
    message: str
    equation: int
    term: int


@dataclass(frozen=True)
class UnknownVariableError(Exception):
    """term.var 引用了未声明的变量名。"""
    message: str
    name: str
    equation: int
    term: int


# ---------------------------------------------------------------------------
# 单位解析：复用表达式解析器，"1" 表示无量纲，拒绝 °C / °F
# ---------------------------------------------------------------------------

def _parse_ratio_unit(expr: str, **where) -> Quantity:
    """解析单位表达式；偏移温标没有唯一的乘性比例，一律拒绝。"""
    if expr.strip() == "1":
        return Quantity((0, 0, 0, 0), Fraction(1))
    try:
        quantity = parse_expression(expr)
    except UnitSyntaxError as exc:
        raise UnitError(exc.message, "syntax_error", exc.position, **where)
    except OffsetScaleInCompoundError as exc:
        raise UnitError(str(exc), "offset_temperature_not_allowed",
                        exc.position, **where)
    if quantity.offset_unit is not None:
        raise UnitError(
            f"offset temperature scale {quantity.offset_unit!r} is not "
            "allowed; use K", "offset_temperature_not_allowed",
            quantity.pos, **where)
    return quantity


# ---------------------------------------------------------------------------
# 输入解析与量纲核对：构建基本单位下的增广矩阵
# ---------------------------------------------------------------------------

def _prepare(variables: list[VariableInput],
             equations: list[EquationInput]
             ) -> tuple[list[tuple[str, Quantity]], list[list[Fraction]]]:
    """解析全部输入并逐项核对量纲，返回 (变量及其单位, 增广矩阵行)。"""
    parsed_vars: list[tuple[str, Quantity]] = []
    for index, variable in enumerate(variables):
        quantity = _parse_ratio_unit(variable.unit, variable=index)
        parsed_vars.append((variable.name, quantity))
    column = {name: i for i, (name, _) in enumerate(parsed_vars)}
    n_vars = len(parsed_vars)

    rows: list[list[Fraction]] = []
    for eq_index, equation in enumerate(equations):
        terms: list[tuple[int, Fraction, Quantity]] = []
        for term_index, term in enumerate(equation.terms):
            col = column.get(term.var)
            if col is None:
                raise UnknownVariableError(
                    f"unknown variable {term.var!r}: not declared in "
                    '"variables"', term.var, eq_index, term_index)
            try:
                coefficient = parse_value(term.coefficient)
            except InvalidValueError as exc:
                raise InvalidNumberError(str(exc), eq_index, term_index)
            quantity = _parse_ratio_unit(term.unit, equation=eq_index,
                                         term=term_index)
            terms.append((col, coefficient, quantity))

        try:
            rhs_value = parse_value(equation.rhs.value)
        except InvalidValueError as exc:
            raise InvalidNumberError(str(exc), eq_index, None)
        rhs = _parse_ratio_unit(equation.rhs.unit, equation=eq_index)

        row = [Fraction(0)] * (n_vars + 1)
        row[n_vars] = rhs_value * rhs.scale
        for term_index, (col, coefficient, quantity) in enumerate(terms):
            var_dims = parsed_vars[col][1].dims
            dims = tuple(a + b for a, b in zip(quantity.dims, var_dims))
            if dims != rhs.dims:
                raise TermDimensionError(
                    "dimension mismatch: coefficient unit times variable "
                    f"unit has {dim_description(dims)} but the right-hand "
                    f"side has {dim_description(rhs.dims)}",
                    eq_index, term_index)
            row[col] += coefficient * quantity.scale
        rows.append(row)
    return parsed_vars, rows


# ---------------------------------------------------------------------------
# 有理系数线性方程组（Gauss-Jordan 精确消元）
# ---------------------------------------------------------------------------

def _eliminate(rows: list[list[Fraction]], n_vars: int,
               variables: list[str]) -> list[Fraction]:
    """对增广矩阵 [A | b] 做 Gauss-Jordan 精确消元，返回唯一解。

    系数部分全零而增广列非零（0 = 非零）即矛盾，抛 NoSolutionError；
    列秩不足（存在自由变量）即欠定，抛 UnderdeterminedError。
    """
    pivot_row: dict[int, int] = {}  # 主元列 -> 行下标
    n_rows = len(rows)
    for col in range(n_vars):
        pivot = next((r for r in range(n_rows)
                      if r not in pivot_row.values() and rows[r][col] != 0),
                     None)
        if pivot is None:
            continue
        pivot_row[col] = pivot
        factor = rows[pivot][col]
        if factor != 1:
            rows[pivot] = [v / factor for v in rows[pivot]]
        for r in range(n_rows):
            k = rows[r][col]
            if r != pivot and k != 0:
                rows[r] = [a - k * b for a, b in zip(rows[r], rows[pivot])]

    used = set(pivot_row.values())
    if any(r not in used and rows[r][n_vars] != 0 for r in range(n_rows)):
        raise NoSolutionError(
            "the system is inconsistent: no solution satisfies all equations")
    free = [c for c in range(n_vars) if c not in pivot_row]
    if free:
        free_variables = sorted(variables[c] for c in free)
        raise UnderdeterminedError(
            "the system is underdetermined; "
            f"{len(free_variables)} variable(s) lack a unique solution: "
            + ", ".join(free_variables),
            free_variables=free_variables)
    return [rows[pivot_row[c]][n_vars] for c in range(n_vars)]


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def solve_quantities(variables: list[VariableInput],
                     equations: list[EquationInput]) -> dict[str, Fraction]:
    """求解带单位线性方程组。

    成功返回 {变量名: 以声明单位表示的精确有理数值}（按变量名排序）；
    失败抛出 UnitError / InvalidNumberError / TermDimensionError /
    UnknownVariableError / SolveError 子类。
    """
    parsed_vars, rows = _prepare(variables, equations)
    n_vars = len(parsed_vars)
    solution = _eliminate(rows, n_vars, [name for name, _ in parsed_vars])
    values = {
        name: solution[i] / quantity.scale
        for i, (name, quantity) in enumerate(parsed_vars)
    }
    return {name: values[name] for name in sorted(values)}
