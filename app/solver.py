"""带单位的线性方程组精确求解。

POST /solve-quantities 的核心计算：

- 1～8 个互异变量各声明单位；1～16 条等式各含非空项列表与右侧。
  项 = 变量引用 × 系数 × 项单位，右侧 = 值 × 单位；
- 逐项核对量纲：dims(项单位) + dims(变量单位) == dims(右侧单位)；
- 系数与右侧值按单位比例换算到基本单位（m/kg/s/K），得到关于
  “变量以其声明单位计的数值”的有理系数线性方程组（同一变量在同一
  等式中多次出现时系数相加）；
- 用 ``Fraction`` 精确做 Gauss-Jordan 消元：出现 0 = 非零 判无解，
  主元列少于变量数判欠定，否则得到唯一解；
- 唯一解即各变量以其声明单位计的数值，按 ROUND_HALF_EVEN 保留最多
  12 位小数的格式化由 :func:`app.converter.format_result` 完成。

``unit="1"`` 仅表示无量纲（比例 1）。``°C`` / ``°F`` 是偏移温标，
没有唯一的乘性比例，变量、项与右侧的单位一律拒绝。
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
    """一个未知变量：name 非空白且互异（请求层校验），unit 为单位表达式。"""
    name: str
    unit: str


@dataclass(frozen=True)
class TermInput:
    """一个项：var 引用已声明变量，coefficient 为有限十进制字符串。"""
    var: str
    coefficient: str
    unit: str


@dataclass(frozen=True)
class RhsInput:
    """等式右侧：value 为有限十进制字符串，unit 为单位表达式。"""
    value: str
    unit: str


@dataclass(frozen=True)
class LinearEquationInput:
    terms: tuple[TermInput, ...]
    rhs: RhsInput


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class SolveError(Exception):
    """求解失败的基类；error 为对外的错误类型标识，extra 为坐标等附加信息。"""

    error = "solve_error"

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


class UnitExpressionError(SolveError):
    """单位表达式无法用于求解。

    kind 为 "syntax_error" / "offset_temperature_not_allowed"；坐标为
    variable（变量下标）或 equation + term（term 缺省表示等式右侧）。
    """

    def __init__(self, kind: str, message: str,
                 position: int | None, **coords):
        super().__init__(message, position=position, **coords)
        self.error = kind


class InvalidNumberError(SolveError):
    """系数或右侧值不是有限十进制字符串；坐标为 equation（+ term）。"""
    error = "invalid_value"


class UnknownVariableError(SolveError):
    """项引用了未声明的变量名；坐标为 equation + term。"""
    error = "unknown_variable"


class TermDimensionMismatchError(SolveError):
    """项单位乘变量单位与右侧单位量纲不一致；坐标为 equation + term。"""
    error = "dimension_mismatch"


class NoSolutionError(SolveError):
    """方程组矛盾，无解。"""
    error = "no_solution"


class UnderdeterminedSystemError(SolveError):
    """方程组欠定，解不唯一；extra 含 free_variables。"""
    error = "underdetermined"


# ---------------------------------------------------------------------------
# 单位与数值解析：复用表达式解析器，"1" 为无量纲，拒绝 °C / °F
# ---------------------------------------------------------------------------

def _parse_unit(expr: str, **coords) -> Quantity:
    """解析单位表达式；"1" 表示无量纲，偏移温标一律拒绝。"""
    if expr.strip() == "1":
        return Quantity((0, 0, 0, 0), Fraction(1))
    try:
        quantity = parse_expression(expr)
    except UnitSyntaxError as exc:
        raise UnitExpressionError(
            "syntax_error", exc.message, exc.position, **coords)
    except OffsetScaleInCompoundError as exc:
        raise UnitExpressionError(
            "offset_temperature_not_allowed", str(exc), exc.position, **coords)
    if quantity.offset_unit is not None:
        raise UnitExpressionError(
            "offset_temperature_not_allowed",
            f"offset temperature scale {quantity.offset_unit!r} is not "
            "allowed; use K", quantity.pos, **coords)
    return quantity


def _parse_number(raw: str, **coords) -> Fraction:
    """把有限十进制字符串解析为精确的有理数。"""
    try:
        return parse_value(raw)
    except InvalidValueError as exc:
        raise InvalidNumberError(str(exc), **coords)


# ---------------------------------------------------------------------------
# 有理系数线性方程组（稠密 Gauss-Jordan 消元）
# ---------------------------------------------------------------------------

def _solve_system(matrix: list[list[Fraction]],
                  rhs: list[Fraction],
                  names: list[str]) -> list[Fraction]:
    """精确求解 A·x = b，返回唯一解（按 names 顺序）。

    矛盾（0 = 非零）抛出 NoSolutionError；主元列少于变量数（存在自由
    变量）抛出 UnderdeterminedSystemError。
    """
    n_vars = len(names)
    # 增广矩阵 [A | b]，原地消元
    rows = [list(row) + [b] for row, b in zip(matrix, rhs)]
    n_rows = len(rows)
    pivot_row_of: dict[int, int] = {}  # 主元列 -> 行
    r = 0
    for c in range(n_vars):
        if r == n_rows:
            break
        pivot = next((i for i in range(r, n_rows) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = 1 / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                k = rows[i][c]
                rows[i] = [a - k * p for a, p in zip(rows[i], rows[r])]
        pivot_row_of[c] = r
        r += 1

    # 消元后非主元行的系数部分全零；增广列非零即矛盾
    for row in rows:
        if all(row[c] == 0 for c in range(n_vars)) and row[n_vars] != 0:
            raise NoSolutionError(
                "the system is inconsistent: no solution exists")

    if len(pivot_row_of) < n_vars:
        free = sorted(names[c] for c in range(n_vars) if c not in pivot_row_of)
        raise UnderdeterminedSystemError(
            "the system is underdetermined; "
            f"{len(free)} variable(s) lack a unique solution: "
            + ", ".join(free),
            free_variables=free)

    # 每列都有主元时，主元行即单位向量，解在增广列
    return [rows[pivot_row_of[c]][n_vars] for c in range(n_vars)]


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def solve_quantities(variables: list[VariableInput],
                     equations: list[LinearEquationInput]
                     ) -> dict[str, Fraction]:
    """求解带单位的线性方程组。

    成功返回 {变量名: 以其声明单位计的数值}（精确有理数）；
    失败抛出 SolveError 子类。
    """
    var_units: list[Quantity] = []
    for index, variable in enumerate(variables):
        var_units.append(_parse_unit(variable.unit, variable=index))
    var_index = {variable.name: i for i, variable in enumerate(variables)}

    n_vars = len(variables)
    matrix: list[list[Fraction]] = []
    rhs_values: list[Fraction] = []

    for eq_index, equation in enumerate(equations):
        rhs_unit = _parse_unit(equation.rhs.unit, equation=eq_index)
        rhs_value = _parse_number(equation.rhs.value, equation=eq_index)

        row = [Fraction(0)] * n_vars
        for term_index, term in enumerate(equation.terms):
            coords = {"equation": eq_index, "term": term_index}
            if term.var not in var_index:
                raise UnknownVariableError(
                    f"unknown variable: {term.var!r}", **coords)
            coefficient = _parse_number(term.coefficient, **coords)
            term_unit = _parse_unit(term.unit, **coords)
            i = var_index[term.var]
            # 量纲核对：项单位 × 变量单位 与右侧单位一致
            term_dims = tuple(a + b for a, b
                              in zip(term_unit.dims, var_units[i].dims))
            if term_dims != rhs_unit.dims:
                raise TermDimensionMismatchError(
                    f"dimension mismatch: term has "
                    f"{dim_description(term_dims)} but the right-hand side "
                    f"has {dim_description(rhs_unit.dims)}", **coords)
            # 换算到基本单位：c · scale(项单位) · scale(变量单位) · X_var
            row[i] += coefficient * term_unit.scale * var_units[i].scale
        matrix.append(row)
        rhs_values.append(rhs_value * rhs_unit.scale)

    solution = _solve_system(matrix, rhs_values,
                             [variable.name for variable in variables])
    return {variable.name: solution[i]
            for i, variable in enumerate(variables)}
