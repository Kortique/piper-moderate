"""Partition the underage SigLIP labels into three aggregate groups so the
model can distinguish "actual child in image" from "child-adjacent context".

Three groups:
- BODY: explicit child anatomy/nudity/sex-act labels. Strong evidence of an
  actual child being depicted.
- CONTEXT: clothing, scene, posing, demographic cues. Can fire on child OR on
  contextual cues without an actual child (classrooms, playgrounds, etc.).
- INTERACTION: `:x`-multiplied "adult with child" labels. These fire (weakly)
  on adult-only size-contrast images too.

Each group gets its own noisy-OR aggregate. Model can then learn:
- High BODY + (any context/interaction) → CSAM
- High CONTEXT only + low BODY + high _adult → probably contextual cue, not CSAM
"""

# Explicit body / anatomy / sex-act labels — high specificity for "actual child in image"
BODY_LABELS = frozenset([
    "baby", "infant", "toddler", "toddler_literal", "toddler_group",
    "preschooler", "kid", "child", "small_child", "little_boy", "little_girl",
    "young_boy", "young_girl", "preteen_boy", "preteen_girl", "preteen_slim",
    "preteen_model", "preteen_smile", "tween", "teen_girl",
    "chubby_cheeks", "child_proportions", "smooth_child_skin",
    "child_genitals", "girl_genitals", "explicit_genitals", "explicit_spread",
    "naked_child", "naked_minor", "naked_teen", "flat_chest_child",
    "flat_chest_nude", "flat_chest_teen", "frail_body",
    "fluids_face", "cum_child_face", "fluids_macro_face", "face_fluids_macro",
    "juvenile_face_fluids", "fluids_undeveloped_body",
    "fragile_crawling", "fragile_boy_anatomy",
    "petite_side_profile", "slender_childish_frame", "slender_minor_group",
    "youthful_face_flat_body", "explicit_expression_juvenile",
    "lying_down_nude_girl", "kneeling_submissive", "kneeling_teen_girl",
    "intercourse_scene_with_girl", "intercourse_boy_body", "two_girls_explicit",
    "shota_bed_scene", "boy_explicit_size", "boy_macro_breasts",
    "boy_overpowered_breasts", "macro_monster_anatomy", "monster_explicit",
    "interracial_flat_anatomy", "cosplay_minor_explicit", "cosplay_bodysuit_minor",
    "girl_multiple_adults", "anime_child", "anime_loli", "anime_explicit",
    "3d_doll_child", "3d_doll_explicit", "3d_preteen_model", "explicit_3d_minor",
    "digital_art_loli", "stylized_big_eyes",
    "explicit_universal_anchor", "universal_anchor",
    "group_naked_teens",
    "bent_over_narrow_hips", "bent_over_strict",
    "pov_face_close_up", "pov_kneeling",
])

# Scene / clothing / pose / demographic CONTEXT labels — can fire without an actual child
CONTEXT_LABELS = frozenset([
    "school", "kindergarten", "playground", "classroom", "child_in_car",
    "child_bedroom", "seatbelt", "stroller", "crib", "sandbox", "toys",
    "pacifier", "bathing_child", "child_swimsuit", "child_swimwear_model",
    "preteen_swimwear", "girl_swimsuit_posing", "girl_bikini",
    "flat_chest_swimwear_portrait", "swimsuit",
    "girl_in_dress", "girl_pretty_dress", "minor_tight_dress", "girl_portrait",
    "girl_posing_photo", "preteen_posing", "teen_selfie", "child_portrait",
    "teen_girl_casual", "teen_girl_sporty", "teen_school",
    "denim_casual_fashion", "casual_bedroom_pose", "bedroom_pose",
    "panties_casual", "crop_top_panties", "bare_legs_indoor",
    "fitness_body_underwear", "childish_panties_bows", "revealing_child_clothes",
    "cosplay", "gymnastics_leotard", "stripping_clothes",
    "glasses_girl_nude", "glasses_child_person",
    "teen_acne", "pigtails", "child_bangs", "child_freckles", "curly_child",
    "short_hair_boy", "missing_teeth", "bridal_child",
    "group", "group_of_children", "children_party", "children_playing",
    "classroom_children", "extreme_size_difference", "giant_adult_small_child",
])

# `:x`-multiplied INTERACTION labels — fire on adult-child compositional cues
INTERACTION_LABELS = frozenset([
    "adult_with_child", "adult_with_girl", "adult_with_boy",
    "adult_child_together", "adult_child_sitting",
    "man_with_young_girl", "man_with_young_boy",
    "woman_with_young_girl", "woman_with_young_boy",
    "older_person_with_child", "maternal_contrast",
    "boy_mature_woman",
])


def categorize(feature_name: str) -> str:
    """Return 'body', 'context', 'interaction', or 'other'."""
    if feature_name in BODY_LABELS:
        return "body"
    if feature_name in CONTEXT_LABELS:
        return "context"
    if feature_name in INTERACTION_LABELS:
        return "interaction"
    return "other"
