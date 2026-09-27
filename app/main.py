"""单位换算 API。

POST /convert 接收 {"value": "...", "from": "...", "to": "..."}，
返回 {"result": "..."}；错误一律返回 422。

POST /infer-dimensions 接收多条乘积等式 {"equations": [...]}，推断未知
变量的（长度、质量、时间、温度）四维量纲；矛盾、欠定或分数解返回 422。

POST /dimensionless-group 接收 1～8 个具名量 {"quantities": [{name, unit}]}，
搜索每项 ∈ [-6, 6] 的最简无量纲乘积指数；无解或输入非法返回 422。

POST /propagate-product 接收 1～8 个独立测量值 {"factors": [...], "to": "..."}，
计算乘积 y=∏x_i^power 及其一阶合成标准不确定度；非法输入返回 422。

POST /solve-quantities 接收 1～8 个互异变量 {"variables": [...]} 与 1～16 条
带单位的线性等式 {"equations": [...]}，逐项核对量纲后用有理数精确消元；
仅唯一解返回按变量名排序的 values，无解、欠定或输入非法返回 422。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .converter import (
    DimensionMismatchError,
    InvalidValueError,
    OffsetScaleInCompoundError,
    Quantity,
    UnitSyntaxError,
    convert_value,
    format_result,
    format_uncertainty,
    parse_expression,
    parse_value,
)
from .dimensionless import NoDimensionlessGroupError, find_dimensionless_group
from .dimensions import (
    DimensionInferenceError,
    EquationInput,
    FactorInput,
    UnitFactorError,
    infer_dimensions,
    parse_unit_dimensions,
)
from .propagation import (
    MeasurementInput,
    MeasurementInputError,
    UnitError,
    ZeroNegativePowerError,
    parse_target_unit,
    propagate_product,
)
from .solver import (
    LinearEquationInput,
    RhsInput,
    SolveError,
    TermInput,
    VariableInput,
    solve_quantities,
)

app = FastAPI(
    title="Unit Converter API",
    version="1.4.0",
    description="长度 / 质量 / 时间 / 温度四维量纲的单位换算，"
                "支持复合单位表达式、°C / °F 偏移温标、量纲推断、"
                "最简无量纲乘积搜索、乘积一阶不确定度传播与"
                "带单位线性方程组精确求解。",
)


class ConvertRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: str = Field(max_length=10_000, description="有限十进制字符串")
    from_: str = Field(alias="from", max_length=2_000, description="源单位表达式")
    to: str = Field(max_length=2_000, description="目标单位表达式")


class ConvertResponse(BaseModel):
    result: str


class FactorModel(BaseModel):
    """乘积等式中的一个因子：var 与 unit 恰有一个，power ∈ [-6, 6]，默认 1。"""
    model_config = ConfigDict(extra="forbid")

    var: str | None = Field(default=None, min_length=1, max_length=64)
    unit: str | None = Field(default=None, min_length=1, max_length=2_000)
    power: int = Field(default=1, ge=-6, le=6)

    @model_validator(mode="after")
    def _exactly_one_of_var_or_unit(self) -> "FactorModel":
        if (self.var is None) == (self.unit is None):
            raise ValueError('exactly one of "var" or "unit" is required')
        if self.var is not None and not self.var.strip():
            raise ValueError('"var" must not be blank')
        return self


class EquationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: list[FactorModel] = Field(min_length=1, max_length=50)
    right: list[FactorModel] = Field(min_length=1, max_length=50)


class InferDimensionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equations: list[EquationModel] = Field(min_length=1, max_length=100)


class DimensionVector(BaseModel):
    length: int
    mass: int
    time: int
    temperature: int


class InferDimensionsResponse(BaseModel):
    dimensions: dict[str, DimensionVector]


class QuantityModel(BaseModel):
    """一个具名量：name 非空白且互异（在请求层校验），unit 为单位表达式。"""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    unit: str = Field(min_length=1, max_length=2_000)


class DimensionlessGroupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantities: list[QuantityModel] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _names_non_blank_and_distinct(self) -> "DimensionlessGroupRequest":
        seen: set[str] = set()
        for quantity in self.quantities:
            if not quantity.name.strip():
                raise ValueError('"name" must not be blank')
            if quantity.name in seen:
                raise ValueError(f'duplicate "name": {quantity.name!r}')
            seen.add(quantity.name)
        return self


class DimensionlessGroupResponse(BaseModel):
    exponents: dict[str, int]


class MeasurementModel(BaseModel):
    """一个独立测量值：value/uncertainty 为有限十进制字符串，
    unit 为单位表达式，power ∈ [-6, 6]（默认 1）。"""
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=10_000, description="有限十进制字符串")
    uncertainty: str = Field(max_length=10_000, description="非负有限十进制字符串")
    unit: str = Field(min_length=1, max_length=2_000, description="单位表达式")
    power: int = Field(default=1, ge=-6, le=6)


class PropagateProductRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factors: list[MeasurementModel] = Field(min_length=1, max_length=8)
    to: str = Field(min_length=1, max_length=2_000, description="目标单位表达式；"
                                                                '"1" 仅表示无量纲')


class PropagateProductResponse(BaseModel):
    value: str
    uncertainty: str


class VariableModel(BaseModel):
    """一个未知变量：name 非空白且互异（在请求层校验），unit 为单位表达式
    （"1" 表示无量纲）。"""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    unit: str = Field(min_length=1, max_length=2_000)


class TermModel(BaseModel):
    """一个项：var 引用已声明变量，coefficient 为有限十进制字符串，
    unit 为单位表达式。"""
    model_config = ConfigDict(extra="forbid")

    var: str = Field(min_length=1, max_length=64)
    coefficient: str = Field(max_length=10_000, description="有限十进制字符串")
    unit: str = Field(min_length=1, max_length=2_000)


class RhsModel(BaseModel):
    """等式右侧：value 为有限十进制字符串，unit 为单位表达式。"""
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=10_000, description="有限十进制字符串")
    unit: str = Field(min_length=1, max_length=2_000)


class SolveEquationModel(BaseModel):
    """一条线性等式：非空项列表 + 右侧。"""
    model_config = ConfigDict(extra="forbid")

    terms: list[TermModel] = Field(min_length=1)
    rhs: RhsModel


class SolveQuantitiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variables: list[VariableModel] = Field(min_length=1, max_length=8)
    equations: list[SolveEquationModel] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _names_non_blank_and_distinct(self) -> "SolveQuantitiesRequest":
        seen: set[str] = set()
        for variable in self.variables:
            if not variable.name.strip():
                raise ValueError('"name" must not be blank')
            if variable.name in seen:
                raise ValueError(f'duplicate "name": {variable.name!r}')
            seen.add(variable.name)
        return self


class SolveQuantitiesResponse(BaseModel):
    values: dict[str, str]


def _unprocessable(error: str, message: str, **extra) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"error": error, "message": message, **extra},
    )


def _parse_unit_field(expr: str, field: str) -> Quantity:
    try:
        return parse_expression(expr)
    except UnitSyntaxError as exc:
        raise _unprocessable(
            "syntax_error", exc.message, field=field, position=exc.position)
    except OffsetScaleInCompoundError as exc:
        raise _unprocessable(
            "offset_temperature_in_compound", str(exc),
            field=field, position=exc.position)


@app.post("/convert", response_model=ConvertResponse)
def convert_endpoint(req: ConvertRequest) -> ConvertResponse:
    try:
        value = parse_value(req.value)
    except InvalidValueError as exc:
        raise _unprocessable("invalid_value", str(exc))

    source = _parse_unit_field(req.from_, "from")
    target = _parse_unit_field(req.to, "to")

    try:
        result = convert_value(value, source, target)
    except DimensionMismatchError as exc:
        raise _unprocessable("dimension_mismatch", str(exc))

    return ConvertResponse(result=format_result(result))


@app.post("/infer-dimensions", response_model=InferDimensionsResponse)
def infer_dimensions_endpoint(
        req: InferDimensionsRequest) -> InferDimensionsResponse:
    equations = [
        EquationInput(
            left=tuple(FactorInput(var=f.var, unit=f.unit, power=f.power)
                       for f in eq.left),
            right=tuple(FactorInput(var=f.var, unit=f.unit, power=f.power)
                        for f in eq.right),
        )
        for eq in req.equations
    ]

    try:
        dimensions = infer_dimensions(equations)
    except UnitFactorError as exc:
        extra = {"equation": exc.equation, "side": exc.side,
                 "factor": exc.factor}
        if exc.position is not None:
            extra["position"] = exc.position
        raise _unprocessable(exc.kind, exc.message, **extra)
    except DimensionInferenceError as exc:
        raise _unprocessable(exc.error, exc.message, **exc.extra)

    ordered = {
        name: DimensionVector(
            length=vec[0], mass=vec[1], time=vec[2], temperature=vec[3])
        for name, vec in sorted(dimensions.items())
    }
    return InferDimensionsResponse(dimensions=ordered)


@app.post("/dimensionless-group", response_model=DimensionlessGroupResponse)
def dimensionless_group_endpoint(
        req: DimensionlessGroupRequest) -> DimensionlessGroupResponse:
    parsed: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, quantity in enumerate(req.quantities):
        try:
            dims = parse_unit_dimensions(quantity.unit)
        except UnitFactorError as exc:
            extra = {"quantity": index}
            if exc.position is not None:
                extra["position"] = exc.position
            raise _unprocessable(exc.kind, exc.message, **extra)
        parsed.append((quantity.name, dims))

    parsed.sort(key=lambda item: item[0])
    try:
        exponents = find_dimensionless_group([dims for _, dims in parsed])
    except NoDimensionlessGroupError as exc:
        raise _unprocessable("no_dimensionless_group", str(exc))

    return DimensionlessGroupResponse(exponents={
        name: exponent for (name, _), exponent in zip(parsed, exponents)})


@app.post("/propagate-product", response_model=PropagateProductResponse)
def propagate_product_endpoint(
        req: PropagateProductRequest) -> PropagateProductResponse:
    measurements = [
        MeasurementInput(value=f.value, uncertainty=f.uncertainty,
                         unit=f.unit, power=f.power)
        for f in req.factors
    ]
    try:
        target = parse_target_unit(req.to)
        result = propagate_product(measurements, target)
    except UnitError as exc:
        extra: dict = {"field": "to"} if exc.factor is None \
            else {"factor": exc.factor}
        if exc.position is not None:
            extra["position"] = exc.position
        raise _unprocessable(exc.kind, exc.message, **extra)
    except MeasurementInputError as exc:
        raise _unprocessable(exc.kind, exc.message, factor=exc.factor)
    except ZeroNegativePowerError as exc:
        raise _unprocessable("zero_negative_power", str(exc),
                             factor=exc.factor)
    except DimensionMismatchError as exc:
        raise _unprocessable("dimension_mismatch", str(exc))

    return PropagateProductResponse(
        value=format_result(result.value),
        uncertainty=format_uncertainty(result.variance))


@app.post("/solve-quantities", response_model=SolveQuantitiesResponse)
def solve_quantities_endpoint(
        req: SolveQuantitiesRequest) -> SolveQuantitiesResponse:
    variables = [VariableInput(name=v.name, unit=v.unit)
                 for v in req.variables]
    equations = [
        LinearEquationInput(
            terms=tuple(TermInput(var=t.var, coefficient=t.coefficient,
                                  unit=t.unit)
                        for t in eq.terms),
            rhs=RhsInput(value=eq.rhs.value, unit=eq.rhs.unit),
        )
        for eq in req.equations
    ]
    try:
        values = solve_quantities(variables, equations)
    except SolveError as exc:
        raise _unprocessable(exc.error, exc.message, **exc.extra)

    return SolveQuantitiesResponse(values={
        name: format_result(values[name]) for name in sorted(values)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
