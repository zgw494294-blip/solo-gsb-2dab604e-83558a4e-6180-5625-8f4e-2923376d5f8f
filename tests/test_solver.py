"""带单位线性方程组求解核心逻辑测试（不经过 HTTP）。"""

from fractions import Fraction

import pytest

from app.solver import (
    InvalidNumberError,
    LinearEquationInput,
    NoSolutionError,
    RhsInput,
    TermDimensionMismatchError,
    TermInput,
    UnderdeterminedSystemError,
    UnitExpressionError,
    UnknownVariableError,
    VariableInput,
    solve_quantities,
)


def V(name, unit="m") -> VariableInput:
    return VariableInput(name=name, unit=unit)


def T(var, coefficient="1", unit="1") -> TermInput:
    return TermInput(var=var, coefficient=coefficient, unit=unit)


def E(terms, rhs_value="1", rhs_unit="m") -> LinearEquationInput:
    return LinearEquationInput(tuple(terms), RhsInput(rhs_value, rhs_unit))


# ---------------------------------------------------------------- 成功求解

def test_single_equation():
    # 2·x = 10 m
    assert solve_quantities([V("x")], [E([T("x", "2")], "10")]) == {
        "x": Fraction(5)}


def test_two_variable_system():
    # x + y = 10 m；x - y = 2 m
    result = solve_quantities([V("x"), V("y")], [
        E([T("x"), T("y")], "10"),
        E([T("x"), T("y", "-1")], "2"),
    ])
    assert result == {"x": Fraction(6), "y": Fraction(4)}


def test_solution_in_declared_unit():
    # x 以 cm 声明：x[cm]/100 = 2 m → x = 200 cm
    assert solve_quantities([V("x", "cm")], [E([T("x")], "2")]) == {
        "x": Fraction(200)}


def test_term_unit_with_dimensions():
    # 3·(m/s)·t[s] = 6 m → t = 2 s
    result = solve_quantities([V("t", "s")],
                              [E([T("t", "3", "m/s")], "6")])
    assert result == {"t": Fraction(2)}


def test_compound_declared_unit_and_term_unit():
    # v 以 km/h 声明：2·h·v = 10 km → v = 5 km/h
    result = solve_quantities([V("v", "km/h")],
                              [E([T("v", "2", "h")], "10", "km")])
    assert result == {"v": Fraction(5)}


def test_same_variable_merges_coefficients():
    # x + 2·x = 9 m → x = 3
    assert solve_quantities([V("x")], [E([T("x"), T("x", "2")], "9")]) == {
        "x": Fraction(3)}


def test_mixed_variable_units():
    # x[m] + y[cm] = 150 cm；x[m] - y[cm] = 50 cm → x = 1 m，y = 50 cm
    result = solve_quantities([V("x"), V("y", "cm")], [
        E([T("x"), T("y")], "150", "cm"),
        E([T("x"), T("y", "-1")], "50", "cm"),
    ])
    assert result == {"x": Fraction(1), "y": Fraction(50)}


def test_dimensionless_one_units():
    # 全部单位为 "1"：2·x = 8 → x = 4
    result = solve_quantities([V("x", "1")],
                              [E([T("x", "2", "1")], "8", "1")])
    assert result == {"x": Fraction(4)}


def test_temperature_in_kelvin():
    assert solve_quantities([V("T", "K")],
                            [E([T("T", "2")], "600", "K")]) == {
        "T": Fraction(300)}


def test_zero_coefficient_term_contributes_nothing():
    # 0·x + 2·x = 4 m → x = 2
    result = solve_quantities([V("x")], [E([T("x", "0"), T("x", "2")], "4")])
    assert result == {"x": Fraction(2)}


def test_negative_solution():
    assert solve_quantities([V("x")], [E([T("x", "2")], "-8")]) == {
        "x": Fraction(-4)}


def test_decimal_and_exponent_notation():
    # 1.5e2·x = 300 m → x = 2
    result = solve_quantities([V("x")], [E([T("x", "1.5e2")], "3e2")])
    assert result == {"x": Fraction(2)}


def test_overdetermined_but_consistent():
    # x = 1；2·x = 2；3·x = 3（相容的超定组）
    result = solve_quantities([V("x")], [
        E([T("x")], "1"),
        E([T("x", "2")], "2"),
        E([T("x", "3")], "3"),
    ])
    assert result == {"x": Fraction(1)}


def test_fractional_solution_exact():
    # 3·x = 1 m → x = 1/3（精确有理数）
    assert solve_quantities([V("x")], [E([T("x", "3")], "1")]) == {
        "x": Fraction(1, 3)}


# ---------------------------------------------------------------- 无解与欠定

def test_no_solution_contradiction():
    with pytest.raises(NoSolutionError):
        solve_quantities([V("x")], [E([T("x")], "1"), E([T("x")], "2")])


def test_no_solution_zero_row_nonzero_rhs():
    # 0·x = 5
    with pytest.raises(NoSolutionError):
        solve_quantities([V("x")], [E([T("x", "0")], "5")])


def test_underdetermined_free_variable():
    with pytest.raises(UnderdeterminedSystemError) as exc_info:
        solve_quantities([V("x"), V("y")], [E([T("x"), T("y")], "1")])
    assert exc_info.value.extra["free_variables"] == ["y"]


def test_underdetermined_unused_variable():
    with pytest.raises(UnderdeterminedSystemError) as exc_info:
        solve_quantities([V("x"), V("y", "s")], [E([T("x")], "1")])
    assert exc_info.value.extra["free_variables"] == ["y"]


def test_underdetermined_duplicate_equations():
    # 两条等价的方程只提供一个约束
    with pytest.raises(UnderdeterminedSystemError):
        solve_quantities([V("x"), V("y")],
                         [E([T("x"), T("y")], "2"),
                          E([T("x", "2"), T("y", "2")], "4")])


# ---------------------------------------------------------------- 输入错误

def test_unknown_variable():
    with pytest.raises(UnknownVariableError) as exc_info:
        solve_quantities([V("x")], [E([T("z")], "1")])
    assert exc_info.value.extra == {"equation": 0, "term": 0}


def test_invalid_coefficient():
    with pytest.raises(InvalidNumberError) as exc_info:
        solve_quantities([V("x")], [E([T("x", "abc")], "1")])
    assert exc_info.value.extra == {"equation": 0, "term": 0}


def test_invalid_rhs_value():
    with pytest.raises(InvalidNumberError) as exc_info:
        solve_quantities([V("x")], [E([T("x")], "1.2.3")])
    assert exc_info.value.extra == {"equation": 0}


def test_dimension_mismatch_term_unit():
    # 项单位 s × 变量单位 m 与右侧 m 量纲不符
    with pytest.raises(TermDimensionMismatchError) as exc_info:
        solve_quantities([V("x")], [E([T("x", "1", "s")], "1")])
    assert exc_info.value.extra == {"equation": 0, "term": 0}


def test_dimension_mismatch_variable_unit():
    # 变量单位为 s，项单位 "1"，右侧为 m
    with pytest.raises(TermDimensionMismatchError):
        solve_quantities([V("x", "s")], [E([T("x")], "1")])


def test_dimension_mismatch_rhs_dimensionless():
    with pytest.raises(TermDimensionMismatchError):
        solve_quantities([V("x")], [E([T("x")], "1", "1")])


def test_zero_coefficient_still_dimension_checked():
    with pytest.raises(TermDimensionMismatchError):
        solve_quantities([V("x")], [E([T("x", "0", "s")], "1")])


def test_offset_temperature_rejected_everywhere():
    # 变量单位
    with pytest.raises(UnitExpressionError) as exc_info:
        solve_quantities([V("T", "°C")], [E([T("T")], "1", "K")])
    assert exc_info.value.error == "offset_temperature_not_allowed"
    # 项单位（单独与复合）
    for unit in ("°F", "m*°C", "°F^2"):
        with pytest.raises(UnitExpressionError) as exc_info:
            solve_quantities([V("x")], [E([T("x", "1", unit)], "1")])
        assert exc_info.value.error == "offset_temperature_not_allowed", unit
    # 右侧单位
    with pytest.raises(UnitExpressionError) as exc_info:
        solve_quantities([V("T", "K")], [E([T("T")], "1", "°C")])
    assert exc_info.value.error == "offset_temperature_not_allowed"


def test_unit_syntax_error_coordinates():
    with pytest.raises(UnitExpressionError) as exc_info:
        solve_quantities([V("x", "m&")], [E([T("x")], "1")])
    assert exc_info.value.error == "syntax_error"
    assert exc_info.value.extra == {"variable": 0, "position": 1}

    with pytest.raises(UnitExpressionError) as exc_info:
        solve_quantities([V("x")], [E([T("x"), T("x", "1", "kg&")], "1")])
    assert exc_info.value.extra == {"equation": 0, "term": 1, "position": 2}

    with pytest.raises(UnitExpressionError) as exc_info:
        solve_quantities([V("x")], [E([T("x")], "1", "m*")])
    assert exc_info.value.extra == {"equation": 0, "position": 2}
