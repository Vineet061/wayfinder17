import base64
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "trips.db"
MAX_IMAGE_BYTES = 8 * 1024 * 1024

api_key_openai = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model="gpt-4o-mini",api_key=api_key_openai)

# ===========================================================================
# 1. Schemas
# ===========================================================================
class EnvironmentType(str, Enum):
    OUTDOOR_NATURAL = "outdoor_natural"
    OUTDOOR_URBAN = "outdoor_urban"
    SEMI_OUTDOOR = "semi_outdoor"
    INDOOR_HOSPITALITY = "indoor_hospitality"
    INDOOR_CULTURAL = "indoor_cultural"


class VantagePoint(str, Enum):
    AERIAL_DRONE = "aerial_drone"
    ELEVATED_ROOFTOP = "elevated_rooftop"
    EYE_LEVEL = "eye_level"
    STREET_LEVEL = "street_level"
    UNDERGROUND_BASEMENT = "underground_basement"


class LocationEstimate(BaseModel):
    primary: Optional[str] = Field(
        default=None,
        description=(
            "The single most likely SPECIFIC place name, as precise as the image allows "
            "(e.g. 'Gulmarg meadows, Kashmir' — not a state or country). "
            "Set to null ONLY if no specific guess is defensible."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Certainty of the primary identification, based on visible landmarks/signage/terrain."
    )
    candidates: List[str] = Field(
        default_factory=list,
        description=(
            "REQUIRED whenever confidence is 'low' or 'medium'. "
            "List 3-6 REAL, specific, named places whose scenery closely matches the image. "
            "NEVER output vague fallbacks like 'somewhere in India', a bare state, or 'unknown'."
        ),
    )

    @model_validator(mode="after")
    def _require_candidates_when_unsure(self):
        if self.confidence in ("low", "medium") and not self.candidates:
            raise ValueError("candidates must be provided when confidence is not high")
        return self


class SpatialSceneAnalysis(BaseModel):
    """
    Captures the physical space, environmental attributes, and aesthetic atmosphere
    extracted from social media imagery for downstream travel recommendation matching.
    """
    location: LocationEstimate = Field(
        description="Best specific location guess plus lookalike candidates when uncertain."
    )

    environment_type: EnvironmentType = Field(
        description="Broad classification of the physical space"
    )
    landscape_setting: Literal[
        "coastal_beach", "cliffside_ocean", "tropical_jungle", "mountain_valley",
        "urban_skyline", "historic_old_town", "desert_dunes", "countryside_vineyard"
    ] = Field(description="Primary natural or architectural landscape type")
    vantage_point: VantagePoint = Field(
        default=VantagePoint.EYE_LEVEL,
        description="Camera perspective and elevation relative to the scene"
    )

    lighting_atmosphere: Literal[
        "golden_hour", "blue_hour_dusk", "bright_daylight",
        "moody_candlelight", "neon_nightlife", "overcast_misty"
    ] = Field(description="Lighting conditions setting the mood")
    vibe_qualities: List[str] = Field(
        default_factory=list,
        description="Descriptive mood tags, e.g., ['serene', 'bohemian', 'high-energy', 'romantic']"
    )
    crowd_density: Literal[
        "empty_intimate", "sparse", "moderately_active", "packed_bustling"
    ] = Field(description="Perceived social energy and occupancy level in the space")

    prominent_spatial_elements: List[str] = Field(
        default_factory=list,
        description="Key physical fixtures visible, e.g., ['infinity pool', 'cobblestone street']"
    )
    suitable_spatial_activities: List[str] = Field(
        default_factory=list,
        description="Activities the space accommodates, e.g., ['sunset lounging', 'swimming']"
    )

    aesthetic_palette: List[str] = Field(
        default_factory=list,
        description="Dominant visual tones, e.g., ['terracotta & stone', 'lush tropical green']"
    )


class Itinary(BaseModel):
    startLoc: str = Field(description="The user's starting location")
    targetLoc: str = Field(description="The destination location")
    estimatedKM: Optional[str] = Field(default=None, description="Approximate travel distance; null if undeterminable")
    nearestStation: Optional[str] = Field(default=None, description="Nearest train/bus/air station; null if unverified")
    nearestCity: Optional[str] = Field(default=None, description="Nearest major city; null if unverified")
    basicItinary: str = Field(description="A concise day-by-day or step-by-step itinerary")
    howToReach: str = Field(description="Practical travel options and route guidance")
    bestTimeToVisit: str = Field(description="Best season or time of day to visit")
    idealDurationStay: str = Field(description="Recommended duration of stay")


class WorkflowState(TypedDict, total=False):
    img: bytes | str
    media_type: str
    location_hint: str
    location: dict
    startLoc: str
    targetLoc: str
    estimatedKM: Optional[str]
    nearestStation: Optional[str]
    nearestCity: Optional[str]
    basicItinary: str
    howToReach: str
    bestTimeToVisit: str
    idealDurationStay: str
    environment_type: EnvironmentType
    landscape_setting: str
    vantage_point: VantagePoint
    lighting_atmosphere: str
    vibe_qualities: List[str]
    crowd_density: str
    prominent_spatial_elements: List[str]
    suitable_spatial_activities: List[str]
    aesthetic_palette: List[str]


structuredResponse3 = model.with_structured_output(SpatialSceneAnalysis, method="function_calling")
structuredResponse2 = model.with_structured_output(Itinary, method="function_calling")


# ===========================================================================
# 2. Graph nodes
# ===========================================================================
def _resolve_destination(loc) -> str:
    if not isinstance(loc, dict):
        return str(loc) if loc else "Unspecified — needs user confirmation"
    if loc.get("primary"):
        return loc["primary"]
    cands = loc.get("candidates") or []
    return cands[0] if cands else "Unspecified — needs user confirmation"


def spatialFeatures(state: WorkflowState):
    image_data = state["img"]
    media_type = state.get("media_type") or "image/jpeg"

    if isinstance(image_data, bytes):
        image_url = f"data:{media_type};base64,{base64.b64encode(image_data).decode('ascii')}"
    else:
        image_url = image_data
        if not image_url.startswith("data:"):
            image_url = f"data:{media_type};base64,{image_url}"

    hint = state.get("location_hint")
    text = (
        "Analyze the image and identify the location. "
        "Never answer at country or state granularity — if you cannot name the exact "
        "place, set confidence to 'low'/'medium' and list specific similar-looking places "
        "in `candidates`. Do NOT output 'somewhere in India' or a bare state name."
    )
    if hint:
        text += f"\nUser-provided hint (may be vague, verify against the image): {hint}"

    message = HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    result = structuredResponse3.invoke([message])
    return result.model_dump()


def itinerary(state: WorkflowState):
    dest = _resolve_destination(state.get("location"))
    spatial_context = {
        "environment_type": state.get("environment_type"),
        "landscape_setting": state.get("landscape_setting"),
        "vantage_point": state.get("vantage_point"),
        "lighting_atmosphere": state.get("lighting_atmosphere"),
        "vibe_qualities": state.get("vibe_qualities", []),
        "prominent_spatial_elements": state.get("prominent_spatial_elements", []),
        "suitable_spatial_activities": state.get("suitable_spatial_activities", []),
        "aesthetic_palette": state.get("aesthetic_palette", []),
    }
    prompt = f"""
    Create a practical travel itinerary using the following information.

    Starting location: {state.get("startLoc", "Not provided")}
    Destination: {dest}
    Visual scene context: {spatial_context}

    Requirements:
    - Use the starting location and destination exactly when provided.
    - Estimate distance in kilometers, stating clearly that it is approximate; if it
      cannot be determined, leave it null rather than guessing.
    - Identify the nearest station and major city; if insufficient info, leave them null.
    - Explain realistic ways to reach the destination.
    - Recommend the best time to visit and an ideal duration of stay.
    - Build a concise itinerary suited to the scene, its activities, and its vibe.
      Format it as one line per step, each starting with "Day N:" where sensible.
    - Do not invent precise transport schedules, opening hours, or distances.
    """
    result = structuredResponse2.invoke(prompt)
    return result.model_dump()


graph = StateGraph(WorkflowState)
graph.add_node("spatialFeatures", spatialFeatures)
graph.add_node("itinerary", itinerary)

graph.add_edge(START, "spatialFeatures")
graph.add_edge("spatialFeatures", "itinerary")
graph.add_edge("itinerary", END)

workflow = graph.compile()


# ===========================================================================
# 3. Storage
# ===========================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS trips (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    thumb       TEXT,
    payload     TEXT NOT NULL
);
"""


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


with db() as _conn:
    _conn.executescript(SCHEMA)


# ===========================================================================
# 4. API
# ===========================================================================
app = FastAPI(title="Wayfind")

SPATIAL_KEYS = [
    "environment_type", "landscape_setting", "vantage_point", "lighting_atmosphere",
    "vibe_qualities", "crowd_density", "prominent_spatial_elements",
    "suitable_spatial_activities", "aesthetic_palette",
]
ITINERARY_KEYS = [
    "startLoc", "targetLoc", "estimatedKM", "nearestStation", "nearestCity",
    "basicItinary", "howToReach", "bestTimeToVisit", "idealDurationStay",
]


def _shape(state: dict) -> dict:
    """Split the flat graph state into the three groups the UI renders."""
    loc = state.get("location") or {}
    return {
        "location": {
            "primary": loc.get("primary"),
            "confidence": loc.get("confidence"),
            "candidates": loc.get("candidates") or [],
            "resolved": _resolve_destination(loc),
        },
        "scene": {k: state.get(k) for k in SPATIAL_KEYS},
        "itinerary": {k: state.get(k) for k in ITINERARY_KEYS},
    }


def _run(image_bytes: bytes, media_type: str, start_loc: str, hint: str) -> dict:
    state: WorkflowState = {"img": image_bytes, "media_type": media_type}
    if start_loc:
        state["startLoc"] = start_loc
    if hint:
        state["location_hint"] = hint
    return workflow.invoke(state)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    start_loc: str = Form(""),
    location_hint: str = Form(""),
):
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "Upload an image file (JPEG, PNG or WebP).")

    data = await image.read()
    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB. Use a smaller file.")

    try:
        state = _run(data, image.content_type, start_loc.strip(), location_hint.strip())
    except Exception as exc:
        raise HTTPException(502, f"Workflow failed: {exc}")

    result = _shape(state)
    result["id"] = uuid.uuid4().hex[:12]
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["inputs"] = {"start_loc": start_loc.strip(), "location_hint": location_hint.strip()}

    thumb = f"data:{image.content_type};base64,{base64.b64encode(data).decode('ascii')}"
    with db() as conn:
        conn.execute(
            "INSERT INTO trips (id, created_at, thumb, payload) VALUES (?,?,?,?)",
            (result["id"], result["created_at"], thumb, json.dumps(result)),
        )
    return result


@app.get("/api/trips")
def list_trips():
    with db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, thumb, payload FROM trips ORDER BY created_at DESC LIMIT 40"
        ).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload"])
        p["thumb"] = r["thumb"]
        out.append(p)
    return out


@app.delete("/api/trips/{trip_id}", status_code=204)
def delete_trip(trip_id: str):
    with db() as conn:
        cur = conn.execute("DELETE FROM trips WHERE id = ?", (trip_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Trip not found.")
    return None
