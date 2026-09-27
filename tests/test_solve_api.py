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

def test_single_equation():
    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "2", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "2"}}
    assert isinstance(r.json()["values"]["x"], str)


def test_two_by_two_system_with_units():
    # x + 2·(m/s)·y = 10 m；x - (m/s)·y = 2 m → x = 14/3 m，y = 8/3 s
    r = solve([v("x", "m"), v("y", "s")], [
        eq([t("x", "1", "1"), t("y", "2", "m/s")], "10", "m"),
        eq([t("x", "1", "1"), t("y", "-1", "m/s")], "2", "m"),
    ])
    assert r.status_code == 200
    assert r.json() == {"values": {
        "x": "4.666666666667", "y": "2.666666666667"}}


def test_values_sorted_by_name():
    r = solve([v("b", "m"), v("a", "s")], [
        eq([t("b", "1", "1")], "1", "m"),
        eq([t("a", "1", "1")], "2", "s"),
    ])
    assert r.status_code == 200
    assert list(r.json()["values"]) == ["a", "b"]


def test_value_converted_back_to_declared_unit():
    r = solve([v("d", "km")], [eq([t("d", "1", "1")], "3000", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"d": "3"}}


def test_term_and_rhs_unit_scales():
    # 2·cm·r = 1 m（r 无量纲）→ r = 50；3·x = 600 cm → x = 2 m
    r = solve([v("r", "1"), v("x", "m")], [
        eq([t("r", "2", "cm")], "1", "m"),
        eq([t("x", "3", "1")], "600", "cm"),
    ])
    assert r.status_code == 200
    assert r.json() == {"values": {"r": "50", "x": "2"}}


def test_repeated_variable_terms_accumulate():
    r = solve([v("x", "m")],
              [eq([t("x", "2", "1"), t("x", "3", "1")], "10", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "2"}}


def test_overdetermined_consistent_across_units():
    r = solve([v("x", "m")], [
        eq([t("x", "1", "1")], "1", "m"),
        eq([t("x", "1", "1")], "100", "cm"),
        eq([t("x", "2", "1")], "2", "m"),
    ])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "1"}}


def test_one_third_rounding():
    r = solve([v("x", "m")], [eq([t("x", "3", "1")], "1", "m")])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "0.333333333333"}}


def test_rounding_half_even_tie():
    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "0.0000000000025", "m")])
    assert r.json()["values"]["x"] == "0.000000000002"
    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "0.0000000000035", "m")])
    assert r.json()["values"]["x"] == "0.000000000004"


def test_negative_and_scientific_notation():
    r = solve([v("r", "1")], [eq([t("r", "1.5e1", "1")], "-30", "1")])
    assert r.status_code == 200
    assert r.json() == {"values": {"r": "-2"}}


def test_kelvin_and_compound_units():
    r = solve([v("T", "K"), v("F", "kg*m/s^2")], [
        eq([t("T", "1", "1")], "300", "K"),
        eq([t("F", "2", "1")], "10", "(kg*m)/s^2"),
    ])
    assert r.status_code == 200
    assert r.json() == {"values": {"F": "5", "T": "300"}}


def test_eight_variables():
    names = "abcdefgh"
    r = solve([v(name, "m") for name in names],
              [eq([t(name, "1", "1")], str(i + 1), "m")
               for i, name in enumerate(names)])
    assert r.status_code == 200
    assert r.json() == {"values": {
        name: str(i + 1) for i, name in enumerate(names)}}


def test_sixteen_equations():
    r = solve([v("x", "m")],
              [eq([t("x", "1", "1")], "5", "m") for _ in range(16)])
    assert r.status_code == 200
    assert r.json() == {"values": {"x": "5"}}


# ---------------------------------------------------------------- 422：无解与欠定

def test_no_solution():
    r = solve([v("x", "m")], [
        eq([t("x", "1", "1")], "1", "m"),
        eq([t("x", "1", "1")], "2", "m"),
    ])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_solution"


def test_no_solution_across_units():
    r = solve([v("x", "m")], [
        eq([t("x", "1", "1")], "1", "m"),
        eq([t("x", "1", "1")], "101", "cm"),
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


def test_unreferenced_variable_is_underdetermined():
    r = solve([v("x", "m"), v("y", "s")], [eq([t("x", "1", "1")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "underdetermined"
    assert detail["free_variables"] == ["y"]


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
    assert detail["field"] == "rhs"


def test_unknown_variable():
    r = solve([v("x", "m")], [eq([t("z", "1", "1")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_variable"
    assert detail["var"] == "z"
    assert detail["equation"] == 0
    assert detail["term"] == 0


# ---------------------------------------------------------------- 422：单位与量纲

def test_term_dimension_mismatch():
    r = solve([v("x", "m")], [eq([t("x", "1", "s")], "1", "m")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "dimension_mismatch"
    assert detail["equation"] == 0
    assert detail["term"] == 0


def test_rhs_dimension_mismatch():
    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "1", "1")])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "dimension_mismatch"


def test_offset_temperature_rejected():
    r = solve([v("t", "°C")], [eq([t("t", "1", "1")], "1", "K")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["variable"] == 0

    r = solve([v("x", "m")], [eq([t("x", "1", "°F")], "1", "m*K")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["equation"] == 0
    assert detail["term"] == 0

    r = solve([v("x", "m")], [eq([t("x", "1", "1")], "1", "m*°C")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["equation"] == 0
    assert detail["field"] == "rhs"


def test_unit_syntax_error_coordinates():
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
    assert detail["field"] == "rhs"
    assert detail["position"] == 2


# ---------------------------------------------------------------- 422：请求体结构

def test_missing_fields_422():
    assert client.post("/solve-quantities", json={}).status_code == 422
    assert client.post("/solve-quantities",
                       json={"variables": [v("x", "m")]}).status_code == 422
    assert client.post("/solve-quantities", json={
        "equations": [eq([t("x", "1", "1")], "1", "m")]}).status_code == 422


def test_variable_count_bounds():
    equations = [eq([t("x", "1", "1")], "1", "m")]
    assert solve([], equations).status_code == 422
    assert solve([v(f"n{i}", "m") for i in range(9)],
                 equations).status_code == 422


def test_equation_count_bounds():
    variables = [v("x", "m")]
    assert solve(variables, []).status_code == 422
    assert solve(variables, [eq([t("x", "1", "1")], "1", "m")
                             for _ in range(17)]).status_code == 422


def test_empty_terms_rejected():
    r = solve([v("x", "m")], [{"terms": [], "rhs": {"value": "1", "unit": "m"}}])
    assert r.status_code == 422


def test_missing_rhs_rejected():
    r = solve([v("x", "m")], [{"terms": [t("x", "1", "1")]}])
    assert r.status_code == 422


def test_missing_term_fields():
    assert solve([v("x", "m")], [eq([{"coefficient": "1", "unit": "1"}],
                                    "1", "m")]).status_code == 422
    assert solve([v("x", "m")], [eq([{"var": "x", "unit": "1"}],
                                    "1", "m")]).status_code == 422
    assert solve([v("x", "m")], [eq([{"var": "x", "coefficient": "1"}],
                                    "1", "m")]).status_code == 422


def test_blank_and_duplicate_names_rejected():
    equations = [eq([t("x", "1", "1")], "1", "m")]
    assert solve([v("   ", "m")], equations).status_code == 422
    assert solve([v("x", "m"), v("x", "s")], equations).status_code == 422


def test_unknown_field_rejected():
    r = client.post("/solve-quantities", json={
        "variables": [{"name": "x", "unit": "m", "note": "nope"}],
        "equations": [eq([t("x", "1", "1")], "1", "m")]})
    assert r.status_code == 422
    r = client.post("/solve-quantities", json={
        "variables": [v("x", "m")],
        "equations": [{"terms": [t("x", "1", "1")],
                       "rhs": {"value": "1", "unit": "m"},
                       "note": "nope"}]})
    assert r.status_code == 422
