"""
Preprocessing Utilities
Builds natural language descriptions from form attributes and validates input.
"""


VALID_GENDERS = ['Male', 'Female', 'Non-Binary', 'Other', '']
VALID_AGE_GROUPS = ['18-25', '25-30', '30-40', '40-50', '50-60', '60+', '']
VALID_SKIN_TONES = ['Very Fair', 'Fair', 'Light Brown', 'Medium Brown', 'Dark Brown', 'Very Dark', '']
VALID_FACE_SHAPES = ['Oval', 'Round', 'Square', 'Heart', 'Oblong', 'Diamond', '']
VALID_EYE_COLORS = ['Brown', 'Blue', 'Green', 'Hazel', 'Gray', 'Black', 'Amber', '']


def build_description(attributes: dict) -> str:
    """
    Build a natural language description from structured form attributes.

    Args:
        attributes: dict with keys like gender, age_group, hair_style, etc.

    Returns:
        A composed natural language description string.
    """
    parts = []

    # Start with basic demographics
    gender = attributes.get('gender', '').strip()
    age_group = attributes.get('age_group', '').strip()

    if gender and age_group:
        parts.append(f"A {age_group} year old {gender.lower()} person")
    elif gender:
        parts.append(f"A {gender.lower()} person")
    elif age_group:
        parts.append(f"A person aged {age_group}")
    else:
        parts.append("A person")

    # Skin tone
    skin_tone = attributes.get('skin_tone', '').strip()
    if skin_tone:
        parts.append(f"with {skin_tone.lower()} skin")

    # Face shape
    face_shape = attributes.get('face_shape', '').strip()
    if face_shape:
        parts.append(f"and a {face_shape.lower()} shaped face")

    # Eye color
    eye_color = attributes.get('eye_color', '').strip()
    if eye_color:
        parts.append(f"with {eye_color.lower()} eyes")

    # Hair
    hair_style = attributes.get('hair_style', '').strip()
    if hair_style:
        parts.append(f"with {hair_style.lower()} hair")

    # Facial hair
    facial_hair = attributes.get('facial_hair', '').strip()
    if facial_hair and facial_hair.lower() != 'none':
        parts.append(f"and {facial_hair.lower()}")

    # Accessories
    accessories = attributes.get('accessories', '').strip()
    if accessories and accessories.lower() != 'none':
        parts.append(f"wearing {accessories.lower()}")

    # Additional description
    description = attributes.get('description', '').strip()
    if description:
        parts.append(f", {description}")

    result = ' '.join(parts)

    # Clean up punctuation
    result = result.replace(' ,', ',').replace('  ', ' ').strip()
    if not result.endswith('.'):
        result += '.'

    return result


def validate_input(data: dict) -> tuple:
    """
    Validate incoming request data.

    Args:
        data: Request JSON data dictionary.

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not data:
        return False, "No data provided"

    if not isinstance(data, dict):
        return False, "Data must be a JSON object"

    # At least one field should be non-empty
    fields = ['gender', 'age_group', 'hair_style', 'facial_hair',
              'accessories', 'description', 'skin_tone', 'face_shape', 'eye_color']

    has_content = False
    for field in fields:
        val = data.get(field, '').strip()
        if val:
            has_content = True
            break

    if not has_content:
        return False, "Please provide at least one description attribute"

    # Validate specific field values if provided
    gender = data.get('gender', '').strip()
    if gender and gender not in VALID_GENDERS:
        return False, f"Invalid gender: {gender}"

    age_group = data.get('age_group', '').strip()
    if age_group and age_group not in VALID_AGE_GROUPS:
        return False, f"Invalid age group: {age_group}"

    skin_tone = data.get('skin_tone', '').strip()
    if skin_tone and skin_tone not in VALID_SKIN_TONES:
        return False, f"Invalid skin tone: {skin_tone}"

    face_shape = data.get('face_shape', '').strip()
    if face_shape and face_shape not in VALID_FACE_SHAPES:
        return False, f"Invalid face shape: {face_shape}"

    eye_color = data.get('eye_color', '').strip()
    if eye_color and eye_color not in VALID_EYE_COLORS:
        return False, f"Invalid eye color: {eye_color}"

    # Check description length
    description = data.get('description', '')
    if len(description) > 1000:
        return False, "Description too long (max 1000 characters)"

    return True, ""
