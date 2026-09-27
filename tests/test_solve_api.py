"""POST /solve-quantities 的 HTTP API 测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def solve(variables, equations):
    return client.post("/solve-quantities",
                       json={"variables": variables, "equations": equations})


def v(name, unit):
    return {"name": name, "unit": unit}


def t(var, coefficient, unit):
    return {"var": var, "coefficient": coefficient, "unit": unit}


def eq(terms, value, unit):
    return {"terms": terms, "rhs": {"value": value, "unit": unit}}


# ---------------------------------------------------------------- 成功

def test_two_variable_system():
    # x + y = 10 m；x - y = 2 m
    r = solve([v("x", "m"), v("y", "m")], [
        eq([t("x", "1", "1"), t("y", "1", "1")], "10", "m"),
        eq([t("x", "1", "1"), t("y", "-1", "1")], "2", "m"),
    ])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "6", "y": "4"}}
    assert isinstance(r.json()["values"]["x"], str)


def test_values_sorted_by_name():
    r = solve([v("b", "m"), v("a", "m")], [
        eq([t("a", "1", "1")], "1", "m"),
        eq([t("b", "1", "1")], "2", "m"),
    ])
    assert r.status_code == 200
    assert list(r.json()["values"]) == ["a", "b"]


def test_solution_in_declared_unit():
    # x 以 cm 声明：x[cm]/100 = 2 m → x = 200 cm
    r = solve([v("x", "cm")], [eq([t("x", "1", "1")], "2", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "200"}}


def test_term_unit_converted_to_base():
    # 3·(m/s)·t[s] = 6 m → t = 2 s
    r = solve([v("t", "s")], [eq([t("t", "3", "m/s")], "6", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"t": "2"}}


def test_compound_units():
    # v 以 km/h 声明：2·h·v = 10 km → v = 5 km/h
    r = solve([v("v", "km/h")], [eq([t("v", "2", "h")], "10", "km")])
    assert r.status_code == 200
    assert r.json() == {"values": {"v": "5"}}


def test_dimensionless_one_units():
    r = solve([v("x", "1")], [eq([t("x", "2", "1")], "8", "1")])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "4"}}


def test_kelvin_ok():
    r = solve([v("T", "K")], [eq([t("T", "2", "1")], "600", "K")])
    assert r.status_code == 200
    assert r.json() == {"values": {"T": "300"}}


def test_rounding_twelve_decimals():
    # 3·x = 1 m → x = 1/3
    r = solve([v("x", "m")], [eq([t("x", "3", "1")], "1", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "0.333333333333"}}


def test_rounding_half_even_ties():
    for rhs, expected in (("0.0000000000005", "0"),
                          ("0.0000000000015", "0.000000000002"),
                          ("0.0000000000025", "0.000000000002"),
                          ("0.0000000000035", "0.000000000004")):
        r = solve([v("x", "m")], [eq([t("x", "1", "1")], rhs, "m")])
        assert r.status_code == 200, rhs
        assert r.json()["values"]["x"] == expected, rhs


def test_trailing_zeros_stripped():
    # 8·x = 1 m → x = 0.125
    r = solve([v("x", "m")], [eq([t("x", "8", "1")], "1", "m")])
    assert r.json() == {"values": {"x": "0.125"}}


def test_eight_variables_and_sixteen_equations():
    variables = [v(f"x{i}", "m") for i in range(8)]
    equations = ([eq([t(f"x{i}", "1", "1")], str(i), "m") for i in range(8)]
                 + [eq([t("x0", "1", "1")], "0", "m")] * 8)
    r = solve(variables, equations)
    assert r.status_code == 200
    assert r.json()["values"] == {f"x{i}": str(i) for i in range(8)}


# ---------------------------------------------------------------- 422：无解与欠定

def test_no_solution():
    r = solve([v("x", "m")], [
        eq([t("x", "1", "1")], "1", "m"),
        eq([t("x", "1", "1")], "2", "m"),
    ])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_solution"


def test_underdetermined():
    r = solve([v("x", "m"), v("y", "m")],
              [eq([t("x", "1", "1"), t("y", "1", "1")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "underdetermined"
    assert detail["free_variables"] == ["y"]


def test_underdetermined_unused_variable():
    r = solve([v("x", "m"), v("y", "s")], [eq([t("x", "1", "1")], "1", "m")])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "underdetermined"


# ---------------------------------------------------------------- 422：数值与变量引用

def test_invalid_coefficient():
    for bad in ("abc", "", "1.2.3", "nan", "inf", "1e10000"):
        r = solve([v("x", "m")], [eq([t("x", bad, "1")], "1", "m")])
        assert r.status_code == 422, bad
        detail = r.json()["detail"]
        assert detail["error"] == "invalid_value"
        assert detail["equation"] == 0
        assert detail["term"] == 0


def test_invalid_rhs_value():
    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "abc", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "invalid_value"
    assert detail["equation"] == 0


def test_unknown_variable():
    r = solve([v("x", "m")], [eq([t("z", "1", "1")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_variable"
    assert detail["equation"] == 0
    assert detail["term"] == 0


def test_error_coordinates_reported():
    r = solve([v("x", "m")], [
        eq([t("x", "1", "1")], "1", "m"),
        eq([t("x", "1", "1"), t("y", "1", "1")], "2", "m"),
    ])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_variable"
    assert detail["equation"] == 1
    assert detail["term"] == 1


# ---------------------------------------------------------------- 422：单位与量纲

def test_dimension_mismatch():
    r = solve([v("x", "m")], [eq([t("x", "1", "s")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "dimension_mismatch"
    assert detail["equation"] == 0
    assert detail["term"] == 0


def test_dimension_mismatch_variable_unit():
    r = solve([v("x", "s")], [eq([t("x", "1", "1")], "1", "m")])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "dimension_mismatch"


def test_offset_temperature_rejected():
    for unit in ("°C", "°F", "K*°C", "°F^2"):
        r = solve([v("T", unit)], [eq([t("T", "1", "1")], "1", "K")])
        assert r.status_code == 422, unit
        detail = r.json()["detail"]
        assert detail["error"] == "offset_temperature_not_allowed"
        assert detail["variable"] == 0


def test_offset_temperature_in_term_and_rhs():
    r = solve([v("x", "m")], [eq([t("x", "1", "°C")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["equation"] == 0
    assert detail["term"] == 0

    r = solve([v("T", "K")], [eq([t("T", "1", "1")], "1", "°F")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["equation"] == 0


def test_unit_syntax_error_has_coordinates_and_position():
    r = solve([v("x", "m&")], [eq([t("x", "1", "1")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["variable"] == 0
    assert detail["position"] == 1

    r = solve([v("x", "m")], [eq([t("x", "1", "kg&")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["equation"] == 0
    assert detail["term"] == 0
    assert detail["position"] == 2

    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "1", "m*")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["equation"] == 0
    assert detail["position"] == 2


# ---------------------------------------------------------------- 422：请求体结构

def test_missing_fields_422():
    assert client.post("/solve-quantities", json={}).status_code == 422
    assert client.post("/solve-quantities",
                       json={"variables": [v("x", "m")]}).status_code == 422
    assert client.post("/solve-quantities", json={
        "equations": [eq([t("x", "1", "1")], "1", "m")]}).status_code == 422


def test_variable_count_bounds():
    one_eq = [eq([t("x0", "1", "1")], "1", "m")]
    assert solve([], one_eq).status_code == 422
    assert solve([v(f"x{i}", "m") for i in range(9)], one_eq).status_code == 422


def test_equation_count_bounds():
    variables = [v("x", "m")]
    assert solve(variables, []).status_code == 422
    assert solve(variables, [eq([t("x", "1", "1")], "1", "m")] * 17
                 ).status_code == 422


def test_empty_terms_rejected():
    r = solve([v("x", "m")], [{"terms": [], "rhs": {"value": "1", "unit": "m"}}])
    assert r.status_code == 422


def test_duplicate_and_blank_names_rejected():
    two = [eq([t("x", "1", "1")], "1", "m")]
    assert solve([v("x", "m"), v("x", "s")], two).status_code == 422
    assert solve([v("   ", "m")], two).status_code == 422


def test_missing_term_and_rhs_fields():
    assert solve([v("x", "m")], [
        {"terms": [{"var": "x", "unit": "1"}],
         "rhs": {"value": "1", "unit": "m"}}]).status_code == 422
    assert solve([v("x", "m")], [
        {"terms": [{"var": "x", "coefficient": "1", "unit": "1"}]}
    ]).status_code == 422
    assert solve([v("x", "m")], [
        {"terms": [t("x", "1", "1")], "rhs": {"value": "1"}}
    ]).status_code == 422


def test_unknown_field_rejected():
    r = client.post("/solve-quantities", json={
        "variables": [{"name": "x", "unit": "m", "note": "nope"}],
        "equations": [eq([t("x", "1", "1")], "1", "m")]})
    assert r.status_code == 422
    r = client.post("/solve-quantities", json={
        "variables": [v("x", "m")],
        "equations": [eq([t("x", "1", "1")], "1", "m")],
        "note": "nope"})
    assert r.status_code == 422
