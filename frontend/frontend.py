import os
import base64
from math import atan2, cos, radians, sin, sqrt
import re
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Recycling Classifier", page_icon="♻️", layout="centered")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    margin-bottom: 0.5rem;
}

p {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("♻️ Recycling Classifier")
st.caption("Upload an image. The app sends it to your FastAPI backend and displays the prediction.")

API_URL = os.getenv("BACKEND_PREDICT_URL", "http://localhost:8000/predict")
TIMEOUT_SECONDS = 60
RECYCLING_MAP_BASE_URL = "https://recycling-map.ch"
RECYCLING_MAP_LANG = "en"
LOCATION_PRESETS = {
    "Winterthur": {"lat": 47.4988, "lng": 8.7241, "radius_km": 15},
    "Zürich": {"lat": 47.3769, "lng": 8.5417, "radius_km": 18},
    "Basel": {"lat": 47.5596, "lng": 7.5886, "radius_km": 18},
    "Bern": {"lat": 46.9480, "lng": 7.4474, "radius_km": 18},
    "St. Gallen": {"lat": 47.4245, "lng": 9.3767, "radius_km": 18},
    "Luzern": {"lat": 47.0502, "lng": 8.3093, "radius_km": 18},
    "Alle Standorte": None,
}

MATERIAL_LOOKUP_ALIASES = [
    {
        "triggers": ["biological", "organic waste", "organic", "bio"],
        "candidates": ["Biogenous waste", "Bio waste", "Green waste"],
    },
    {
        "triggers": ["trash", "residual waste", "residual", "waste"],
        "candidates": ["Household waste", "Residual"],
    },
    {
        "triggers": ["clothes", "textiles", "textile", "clothing", "garments"],
        "candidates": ["Textiles and Shoes", "Textiles", "Textile"],
    },
]


def make_preview_base64(file_bytes: bytes) -> str:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    image.thumbnail((100, 100))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def normalize_material_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


@st.cache_data(ttl=60 * 60 * 24)
def fetch_recycling_map_materials():
    collectibles_url = f"{RECYCLING_MAP_BASE_URL}/{RECYCLING_MAP_LANG}/collectible/data"
    details_url = f"{RECYCLING_MAP_BASE_URL}/api/collectible/{RECYCLING_MAP_LANG}"

    collectibles_response = requests.get(collectibles_url, timeout=15)
    collectibles_response.raise_for_status()
    collectibles = collectibles_response.json()

    details_response = requests.get(details_url, timeout=15)
    details_response.raise_for_status()
    details = details_response.json()
    details_by_id = {entry.get("id"): entry for entry in details}

    merged = []
    for item in collectibles:
        material_id = item.get("id")
        slug = item.get("slug")
        name = (item.get("name") or "").strip()
        if not material_id or not slug or not name:
            continue

        info = details_by_id.get(material_id, {})
        merged.append(
            {
                "id": material_id,
                "slug": slug,
                "name": name,
                "normalized_name": normalize_material_name(name),
                "info_collect": info.get("info_collect"),
                "info_nocollect": info.get("info_nocollect"),
                "info_important": info.get("info_important"),
            }
        )

    return merged


def lookup_recycling_map_material(material: str):
    normalized_target = normalize_material_name(material)
    if not normalized_target:
        return None

    try:
        materials = fetch_recycling_map_materials()
    except requests.RequestException:
        return None

    exact_match = next(
        (entry for entry in materials if entry["normalized_name"] == normalized_target),
        None,
    )
    if exact_match:
        return exact_match

    for alias_group in MATERIAL_LOOKUP_ALIASES:
        trigger_matches = any(
            normalized_target == normalize_material_name(trigger)
            or normalized_target in normalize_material_name(trigger)
            or normalize_material_name(trigger) in normalized_target
            for trigger in alias_group["triggers"]
        )
        if not trigger_matches:
            continue

        for alias in alias_group["candidates"]:
            normalized_alias = normalize_material_name(alias)
            alias_exact_match = next(
                (entry for entry in materials if entry["normalized_name"] == normalized_alias),
                None,
            )
            if alias_exact_match:
                return alias_exact_match

            alias_contains_match = next(
                (
                    entry
                    for entry in materials
                    if normalized_alias in entry["normalized_name"]
                    or entry["normalized_name"] in normalized_alias
                ),
                None,
            )
            if alias_contains_match:
                return alias_contains_match

    contains_match = next(
        (
            entry
            for entry in materials
            if normalized_target in entry["normalized_name"]
            or entry["normalized_name"] in normalized_target
        ),
        None,
    )
    return contains_match


def make_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "location"


def haversine_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_km = 6371.0
    delta_lat = radians(lat2 - lat1)
    delta_lng = radians(lng2 - lng1)
    a = sin(delta_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lng / 2) ** 2
    return 2 * earth_radius_km * atan2(sqrt(a), sqrt(1 - a))


@st.cache_data(ttl=60 * 60)
def fetch_top_recycling_places(collectible_id: int, limit: int = 3, location_name: str = "Winterthur"):
    if not collectible_id:
        return []

    url = f"{RECYCLING_MAP_BASE_URL}/api/collection-points/data"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    points = response.json()

    location = LOCATION_PRESETS.get(location_name)
    location_lat = location["lat"] if location else None
    location_lng = location["lng"] if location else None
    location_radius_km = location["radius_km"] if location else None

    candidates = []
    for point in points:
        if point.get("active") != 1:
            continue

        materials = point.get("materials") or []
        matched_material = next(
            (material for material in materials if material.get("id") == collectible_id),
            None,
        )
        if not matched_material:
            continue

        point_lat = point.get("lat")
        point_lng = point.get("lng")
        distance_km = None
        if location_lat is not None and location_lng is not None and point_lat is not None and point_lng is not None:
            distance_km = haversine_distance_km(location_lat, location_lng, point_lat, point_lng)
            if location_radius_km is not None and distance_km > location_radius_km:
                continue

        point_id = point.get("id")
        name = (point.get("name") or "Collection point").strip()
        street = (point.get("street") or "").strip()
        score = matched_material.get("importance", 99)
        name_slug = make_slug(name)
        street_slug = make_slug(street) if street else "address"
        detail_url = (
            f"{RECYCLING_MAP_BASE_URL}/{RECYCLING_MAP_LANG}/collection-points/"
            f"{point_id}/{name_slug}/{street_slug}"
        )

        candidates.append(
            {
                "name": name,
                "street": street,
                "zip_more": point.get("zip_more"),
                "url": detail_url,
                "score": score,
                "distance_km": distance_km,
            }
        )

    if location:
        candidates.sort(
            key=lambda x: (
                x["distance_km"] if x["distance_km"] is not None else float("inf"),
                x["score"],
                str(x["zip_more"] or ""),
                x["name"],
            )
        )
    else:
        candidates.sort(
            key=lambda x: (
                x["score"],
                str(x["zip_more"] or ""),
                x["name"],
            )
        )
    return candidates[:limit]


uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

selected_location = st.selectbox(
    "Search location",
    options=list(LOCATION_PRESETS.keys()),
    index=0,
)

st.caption(f"Current backend: `{API_URL}`")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    preview_base64 = make_preview_base64(file_bytes)


    st.markdown(
        f"""
        <div style="text-align:center; margin-top:10px;">
            <img src="data:image/png;base64,{preview_base64}" width="100">
            <div style="color:#777; margin-top:5px;">Preview</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_clicked = st.button(
            "Analyze image",
            type="primary",
            use_container_width=True,
        )

    if analyze_clicked:
        with st.spinner("Sending image to backend and waiting for response..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type or "application/octet-stream",
                    )
                }

                response = requests.post(API_URL, files=files, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                result = response.json()

                material = result.get("material", "Unknown")
                confidence = result.get("confidence")
                more_info = result.get("more_info_url")
                recycle_info = result.get("recycle_url")
                recycling_map_match = lookup_recycling_map_material(material)
                more_information_text = None
                top_recycling_places = []

                if recycling_map_match:
                    material_id = recycling_map_match["id"]
                    slug = recycling_map_match["slug"]
                    more_info = (
                        f"{RECYCLING_MAP_BASE_URL}/{RECYCLING_MAP_LANG}/collected-items/"
                        f"{material_id}-{slug}"
                    )
                    recycle_info = f"{RECYCLING_MAP_BASE_URL}/{RECYCLING_MAP_LANG}/map"
                    info_sections = []
                    if recycling_map_match.get("info_collect"):
                        info_sections.append(f"Accepted: {recycling_map_match['info_collect']}")
                    if recycling_map_match.get("info_nocollect"):
                        info_sections.append(f"Not accepted: {recycling_map_match['info_nocollect']}")
                    if recycling_map_match.get("info_important"):
                        info_sections.append(f"Important: {recycling_map_match['info_important']}")
                    if info_sections:
                        more_information_text = "\n\n".join(info_sections)

                    try:
                        top_recycling_places = fetch_top_recycling_places(
                            material_id,
                            limit=3,
                            location_name=selected_location,
                        )
                    except requests.RequestException:
                        top_recycling_places = []

                st.success("Analysis completed successfully")

                st.subheader("Result")
                st.write(f"**Material:** {material}")

                if confidence is not None:
                    try:
                        st.write(f"**Confidence:** {float(confidence):.2%}")
                    except (TypeError, ValueError):
                        st.write(f"**Confidence:** {confidence}")

                if more_information_text:
                    st.markdown(f"**More information:**\n\n{more_information_text}")
                elif more_info:
                    st.markdown(f"**More information:** {more_info}")

                if top_recycling_places:
                    st.markdown(f"**Where to recycle in {selected_location} (Top 3):**")
                    for index, place in enumerate(top_recycling_places, start=1):
                        location = place["name"]
                        if place.get("street"):
                            location = f"{location}, {place['street']}"
                        st.markdown(f"{index}. [{location}]({place['url']})")
                elif recycle_info:
                    st.markdown(f"**Where to recycle:** [Link]({recycle_info})")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend. Check whether FastAPI is running and the URL is correct."
                )
            except requests.exceptions.Timeout:
                st.error("The request to the backend timed out.")
            except requests.exceptions.HTTPError as exc:
                error_text = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Backend error: {error_text}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
else:
    st.info("Please upload an image to start the analysis.")