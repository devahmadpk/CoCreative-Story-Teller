CONCEPT_ALIASES = {
    "addition": ["addition", "counting on", "repeated addition"],
    "subtraction": ["subtraction", "counting back", "difference between"],
    "division": ["division", "equal shares"],
    "fractions": ["fractions", "partitioning", "fraction of group"],
    "place value": ["place value", "3-digit place value", "tens and ones"],
    "integers": ["integers", "number line integers"],
    "geometry": ["area", "perimeter", "angles"]
}

# Checks user query against known aliases to map to a canonical concept name. If no match, returns the original concept string. This helps standardize concepts for RAG retrieval and evaluation.
def normalize_concept(concept: str):
    concept = concept.lower()

    for key, aliases in CONCEPT_ALIASES.items():
        if concept in aliases:
            return key

    return concept