"""
This file stores helper functions which are used within the general interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import re
from typing import *

# Third Party

# Private


def sanitize_field_name(name: str) -> str:
    sanitized = name.strip().replace(' ', '_')
    sanitized = re.sub(r'\W|^(?=\d)', '', sanitized)
    if sanitized and sanitized[0].isdigit():
        raise ValueError(f"Invalid field name '{name}': After sanitization, field name cannot start with a digit.")
    return sanitized

def generate_model(keywords: List[str], datatypes: List[Type[Any]]) -> Type[Any]:
    sanitized_keywords = [sanitize_field_name(k) for k in keywords]
    fields = {k: (t, ...) for k, t in zip(sanitized_keywords, datatypes)}
    return create_model('DynamicModel', **fields)