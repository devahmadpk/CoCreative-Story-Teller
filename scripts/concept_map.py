CONCEPT_ALIASES = {
    "addition": ["addition", "counting on", "repeated addition"],
    "subtraction": ["subtraction", "counting back", "difference between"],
    "division": ["division", "equal shares"],
    "fractions": ["fractions", "partitioning", "fraction of group"],
    "place value": ["place value", "3-digit place value", "tens and ones"],
    "integers": ["integers", "number line integers"],
    "geometry": ["area", "perimeter", "angles"]
}


def normalize_concept(concept: str):
    concept = concept.lower()

    for key, aliases in CONCEPT_ALIASES.items():
        if concept in aliases:
            return key

    return concept