import os


import pytest
from vlm.constants import OBJECT_TYPES, extract_object_type


@pytest.mark.parametrize("caption,expected", [
    ("a police car is seen in this und photo", "police_car"),
    ("a patrol car driving down the street", "police_car"),
    ("a silver car driving down a street", "vehicle"),
    ("a blue ford f150 truck entering", "vehicle"),
    ("a man walking across a street", "person"),
    ("a person riding a bicycle", "person"),
    ("a bicycle parked near the gate", "bicycle"),
    ("a cyclist on the road", "bicycle"),
    ("a dog running across the parking lot", "animal"),
    ("a cat sitting on the sidewalk", "animal"),
    ("a bird on the fence", "animal"),
    ("a blurry object in the distance", "unknown"),
])
def test_extract_object_type(caption, expected):
    assert extract_object_type(caption) == expected


def test_police_car_not_classified_as_generic_vehicle():
    assert extract_object_type("a police car is seen in this und photo") == "police_car"
    assert extract_object_type("a police car is seen in this und photo") != "vehicle"


def test_object_types_list_is_complete():
    assert "person" in OBJECT_TYPES
    assert "police_car" in OBJECT_TYPES
    assert "bicycle" in OBJECT_TYPES
    assert "animal" in OBJECT_TYPES
    assert "vehicle" in OBJECT_TYPES
    assert "unknown" in OBJECT_TYPES
