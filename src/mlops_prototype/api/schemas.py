from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input features expected by the Bank Marketing model."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=120)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int = Field(ge=0)
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    poutcome: str

    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

    def to_feature_dict(self) -> dict[str, object]:
        """Convert API field names to the original dataset column names."""

        values = self.model_dump()

        column_mapping = {
            "emp_var_rate": "emp.var.rate",
            "cons_price_idx": "cons.price.idx",
            "cons_conf_idx": "cons.conf.idx",
            "nr_employed": "nr.employed",
        }

        return {
            column_mapping.get(name, name): value
            for name, value in values.items()
        }


class PredictionResponse(BaseModel):
    prediction: Literal["yes", "no"]
    probability_yes: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_alias: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    status: str
    registered_model_name: str
    model_alias: str
    model_version: str | None = None
    roc_auc: float | None = None