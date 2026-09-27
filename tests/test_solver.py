"""线性方程组求解核心逻辑测试（不经过 HTTP）。"""

from fractions import Fraction

import pytest

from app.solver import (
    EquationInput,
    InvalidNumberError,
    NoSolutionError,
    RhsInput,
    TermDimensionError,
    TermInput,
    UnderdeterminedError,
    UnitError,
    UnknownVariableError,
    VariableInput,
    solve_quantities,
)


def V(name, unit):
    return VariableInput(name, unit)


def T(var, coefficient, unit):
    return TermInput(var, coefficient, unit)


def E(terms, value, unit):
    return EquationInput(tuple(terms), RhsInput(value, unit))


# ---------------------------------------------------------------- 成功求解

def test_single_equation():
    assert solve_quantities(
        [V("x", "m")], [E([T("x", "1", "1")], "2", "m")]) == {
        "x": Fraction(2)}


def test_two_by_two_system():
    # x + 2·(m/s)·y = 10 m；x - (m/s)·y = 2 m → x = 14/3 m，y = 8/3 s
    result = solve_quantities(
        [V("x", "m"), V("y", "s")],
        [E([T("x", "1", "1"), T("y", "2", "m/s")], "10", "m"),
         E([T("x", "1", "1"), T("y", "-1", "m/s")], "2", "m")])
    assert result == {"x": Fraction(14, 3), "y": Fraction(8, 3)}


def test_result_sorted_by_name():
    result = solve_quantities(
        [V("b", "m"), V("a", "s")],
        [E([T("b", "1", "1")], "1", "m"),
         E([T("a", "1", "1")], "2", "s")])
    assert list(result) == ["a", "b"]


def test_value_converted_back_to_declared_unit():
    # d_base = 3000 m → d = 3 km
    assert solve_quantities(
        [V("d", "km")], [E([T("d", "1", "1")], "3000", "m")]) == {
        "d": Fraction(3)}


def test_term_unit_scale_applied():
    # 2·cm·r = 1 m（r 无量纲）→ r = 50
    assert solve_quantities(
        [V("r", "1")], [E([T("r", "2", "cm")], "1", "m")]) == {
        "r": Fraction(50)}


def test_rhs_unit_scale_applied():
    # 3·x = 600 cm → x = 2 m
    assert solve_quantities(
        [V("x", "m")], [E([T("x", "3", "1")], "600", "cm")]) == {
        "x": Fraction(2)}


def test_repeated_variable_terms_accumulate():
    # 2x + 3x = 10 m → x = 2 m
    assert solve_quantities(
        [V("x", "m")],
        [E([T("x", "2", "1"), T("x", "3", "1")], "10", "m")]) == {
        "x": Fraction(2)}


def test_overdetermined_but_consistent():
    # x = 1 m、x = 100 cm、2x = 2 m 一致 → x = 1 m
    assert solve_quantities(
        [V("x", "m")],
        [E([T("x", "1", "1")], "1", "m"),
         E([T("x", "1", "1")], "100", "cm"),
         E([T("x", "2", "1")], "2", "m")]) == {
        "x": Fraction(1)}


def test_dimensionless_and_negative_values():
    # 1.5e1·r = -30 → r = -2
    assert solve_quantities(
        [V("r", "1")], [E([T("r", "1.5e1", "1")], "-30", "1")]) == {
        "r": Fraction(-2)}


def test_kelvin_and_compound_units():
    result = solve_quantities(
        [V("T", "K"), V("F", "kg*m/s^2")],
        [E([T("T", "1", "1")], "300", "K"),
         E([T("F", "2", "1")], "10", "kg*m/s^2")])
    assert result == {"F": Fraction(5), "T": Fraction(300)}


# ---------------------------------------------------------------- 无解与欠定

def test_no_solution():
    with pytest.raises(NoSolutionError):
        solve_quantities(
            [V("x", "m")],
            [E([T("x", "1", "1")], "1", "m"),
             E([T("x", "1", "1")], "2", "m")])


def test_no_solution_error_type():
    try:
        solve_quantities(
            [V("x", "m")],
            [E([T("x", "1", "1")], "1", "m"),
             E([T("x", "1", "1")], "101", "cm")])
    except NoSolutionError as exc:
        assert exc.error == "no_solution"
    else:
        raise AssertionError("expected NoSolutionError")


def test_underdetermined_reports_free_variables():
    with pytest.raises(UnderdeterminedError) as info:
        solve_quantities(
            [V("x", "m"), V("y", "m")],
            [E([T("x", "1", "1"), T("y", "1", "1")], "1", "m")])
    assert info.value.error == "underdetermined"
    assert info.value.extra["free_variables"] == ["y"]


def test_unreferenced_variable_is_underdetermined():
    with pytest.raises(UnderdeterminedError) as info:
        solve_quantities(
            [V("x", "m"), V("y", "s")],
            [E([T("x", "1", "1")], "1", "m")])
    assert info.value.extra["free_variables"] == ["y"]


def test_inconsistent_beats_underdetermined():
    # 同时矛盾与欠定：优先报矛盾
    with pytest.raises(NoSolutionError):
        solve_quantities(
            [V("x", "m"), V("y", "m")],
            [E([T("x", "1", "1")], "1", "m"),
             E([T("x", "1", "1")], "2", "m")])


# ---------------------------------------------------------------- 输入错误

def test_unknown_variable():
    with pytest.raises(UnknownVariableError) as info:
        solve_quantities(
            [V("x", "m")], [E([T("z", "1", "1")], "1", "m")])
    exc = info.value
    assert exc.name == "z"
    assert (exc.equation, exc.term) == (0, 0)


def test_term_dimension_mismatch():
    with pytest.raises(TermDimensionError) as info:
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "s")], "1", "m")])
    assert (info.value.equation, info.value.term) == (0, 0)


def test_rhs_dimension_mismatch():
    with pytest.raises(TermDimensionError):
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "1")], "1", "1")])


def test_invalid_coefficient():
    with pytest.raises(InvalidNumberError) as info:
        solve_quantities(
            [V("x", "m")], [E([T("x", "abc", "1")], "1", "m")])
    assert (info.value.equation, info.value.term) == (0, 0)


def test_invalid_rhs_value():
    with pytest.raises(InvalidNumberError) as info:
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "1")], "nan", "m")])
    assert (info.value.equation, info.value.term) == (0, None)


def test_unit_syntax_error_coordinates():
    with pytest.raises(UnitError) as info:
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "kg&")], "1", "m")])
    exc = info.value
    assert exc.kind == "syntax_error"
    assert (exc.equation, exc.term, exc.position) == (0, 0, 2)

    with pytest.raises(UnitError) as info:
        solve_quantities([V("x", "m&")], [])
    assert info.value.variable == 0

    with pytest.raises(UnitError) as info:
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "1")], "1", "m*")])
    exc = info.value
    assert (exc.equation, exc.term) == (0, None)


def test_offset_temperature_rejected_everywhere():
    with pytest.raises(UnitError) as info:
        solve_quantities([V("t", "°C")], [])
    assert info.value.kind == "offset_temperature_not_allowed"

    with pytest.raises(UnitError):
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "°F")], "1", "m*K")])

    with pytest.raises(UnitError):
        solve_quantities(
            [V("x", "m")], [E([T("x", "1", "1")], "1", "m*°C")])
