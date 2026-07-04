"""BLIP caption parsing — object types and keyword taxonomies."""

OBJECT_TYPES = (
    "person",
    "police_car",
    "bicycle",
    "animal",
    "vehicle",
    "unknown",
)

# Matched before generic vehicle (captions like "police car" also contain "car")
POLICE_CAR_KEYWORDS = frozenset({
    "police car", "police vehicle", "patrol car", "cop car",
    "police suv", "police truck", "sheriff", "police cruiser",
})

BICYCLE_KEYWORDS = frozenset({
    "bicycle", "bike", "bicyclist", "cyclist", "mountain bike",
    "road bike", "bmx", "ebike", "e-bike",
})

ANIMAL_KEYWORDS = frozenset({
    "dog", "cat", "bird", "animal", "horse", "cow", "deer",
    "squirrel", "raccoon", "fox", "pet", "puppy", "kitten",
    "goat", "sheep", "bear", "coyote",
})

PERSON_KEYWORDS = frozenset({
    "person", "people", "man", "woman", "human", "pedestrian",
    "figure", "individual", "child", "boy", "girl",
})

VEHICLE_KEYWORDS = frozenset({
    "car", "truck", "vehicle", "van", "suv", "bus", "motorcycle", "sedan", "pickup",
    "crossover", "hatchback", "coupe", "minivan", "jeep", "concept",
    "toyota", "honda", "ford", "nissan", "bmw", "audi", "hyundai",
    "volkswagen", "chevrolet", "mercedes", "kia", "mazda", "tata", "suzuki",
})

COLOR_KEYWORDS = frozenset({
    "red", "blue", "green", "white", "black", "grey", "gray",
    "yellow", "silver", "brown", "orange", "teal", "purple", "maroon", "gold",
})

# Used by alert rules and vehicle tracker
MOTOR_VEHICLE_TYPES = frozenset({"vehicle", "police_car"})


def _contains_keyword(text: str, keywords: frozenset) -> bool:
    lower = text.lower()
    return any(k in lower for k in keywords)


def extract_object_type(caption: str) -> str:
    """Map a BLIP caption to a structured object type (first match wins)."""
    lower = caption.lower()

    if _contains_keyword(lower, POLICE_CAR_KEYWORDS):
        return "police_car"
    if _contains_keyword(lower, PERSON_KEYWORDS):
        return "person"
    if _contains_keyword(lower, BICYCLE_KEYWORDS):
        return "bicycle"
    if _contains_keyword(lower, ANIMAL_KEYWORDS):
        return "animal"
    if _contains_keyword(lower, VEHICLE_KEYWORDS):
        return "vehicle"
    return "unknown"


def extract_color(caption: str) -> str:
    lower = caption.lower()
    for color in COLOR_KEYWORDS:
        if color in lower:
            return color
    return ""
