from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


class GeometryType(str, Enum):
    POINT = "Point"
    LINESTRING = "LineString"


class GeoJSONGeometry(BaseModel):
    type: GeometryType
    coordinates: List


class FeaturePropertiesBase(BaseModel):
    id: str
    name: str = ""
    active: bool = True


class PointSourceProperties(FeaturePropertiesBase):
    lwa: float = Field(..., description="A-weighted sound power level")
    octave_lw: Optional[Dict[int, float]] = None


class LineSourceProperties(FeaturePropertiesBase):
    lwa_per_m: float = Field(..., description="A-weighted sound power per meter")
    octave_lw_per_m: Optional[Dict[int, float]] = None


class BarrierProperties(FeaturePropertiesBase):
    height: Optional[float] = None
    base_elevation: Optional[float] = None


class SectionProperties(FeaturePropertiesBase):
    step_s: float = 25.0
    z_min: float = 0.0
    z_max: float = 50.0
    z_step: float = 2.0


class Feature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[Feature] = Field(default_factory=list)


class BackgroundType(str, Enum):
    GEOREF = "georef"
    CALIBRATED = "calibrated"


class BackgroundConfig(BaseModel):
    kind: BackgroundType
    path: str
    opacity: float = 0.6
    visible: bool = True
    locked: bool = False
    affine: Optional[List[float]] = None
    bbox: Optional[List[float]] = None


class DEMConfig(BaseModel):
    path: str
    nodata: Optional[float] = None
    clamp_sources: bool = False


class CalculationSettings(BaseModel):
    extent: List[float] = Field(default_factory=lambda: [0, 0, 1000, 1000])
    resolution: float = 25.0
    receiver_height: float = 1.5
    ground_factor: float = 0.5
    humidity: float = 70.0
    temperature_c: float = 20.0
    pressure_hpa: float = 1013.25


class ProjectModel(BaseModel):
    schema_version: str = "1.0"
    name: str = "rumore-project"
    crs_epsg: int = 32633
    background: Optional[BackgroundConfig] = None
    dem: Optional[DEMConfig] = None
    point_sources: FeatureCollection = Field(default_factory=FeatureCollection)
    line_sources: FeatureCollection = Field(default_factory=FeatureCollection)
    barriers: FeatureCollection = Field(default_factory=FeatureCollection)
    sections: FeatureCollection = Field(default_factory=FeatureCollection)
    settings: CalculationSettings = Field(default_factory=CalculationSettings)

    @model_validator(mode="after")
    def validate_extent(self) -> "ProjectModel":
        x_min, y_min, x_max, y_max = self.settings.extent
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid extent: max must be greater than min")
        return self


class ScenarioUpdateRequest(BaseModel):
    active_source_ids: List[str]


class SectionRequest(BaseModel):
    section_feature_id: str


class CalculationResponse(BaseModel):
    scenario_raster: str
    isophones_geojson: str
    cache_id: str


class ContributionResponse(BaseModel):
    source_id: str
    raster: str
