"""Frozen benchmark definitions reused by E0, E1 and the source audit tools.

The original protocol file stays at configs/experiment0.json because published
artifacts bind its bytes. Reuse that snapshot, not a second mutable copy.
"""

from pathlib import Path

from .io import read_json


MMLU_NONOVERLAP_EXCLUDED_GROUPS = {
    "Bio / medicine": (
        "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "high_school_biology", "human_aging", "medical_genetics", "nutrition",
        "professional_medicine", "virology",
    ),
    "Chemistry": ("college_chemistry", "high_school_chemistry"),
    "Cyber / computer science": (
        "college_computer_science", "computer_security", "high_school_computer_science",
    ),
}
MMLU_NONOVERLAP_EXCLUDED_SUBJECTS = frozenset(
    subject for subjects in MMLU_NONOVERLAP_EXCLUDED_GROUPS.values() for subject in subjects
)
MMLU_STANDARD_SUBJECTS = frozenset(
    """
    abstract_algebra anatomy astronomy business_ethics clinical_knowledge college_biology
    college_chemistry college_computer_science college_mathematics college_medicine college_physics
    computer_security conceptual_physics econometrics electrical_engineering elementary_mathematics
    formal_logic global_facts high_school_biology high_school_chemistry high_school_computer_science
    high_school_european_history high_school_geography high_school_government_and_politics
    high_school_macroeconomics high_school_mathematics high_school_microeconomics high_school_physics
    high_school_psychology high_school_statistics high_school_us_history high_school_world_history
    human_aging human_sexuality international_law jurisprudence logical_fallacies machine_learning
    management marketing medical_genetics miscellaneous moral_disputes moral_scenarios nutrition
    philosophy prehistory professional_accounting professional_law professional_medicine
    professional_psychology public_relations security_studies sociology us_foreign_policy virology
    world_religions
    """.split()
)


def load_frozen_config(code_dir: Path) -> dict:
    """Read the shared model/dataset revisions from the original frozen snapshot."""
    return read_json(Path(code_dir) / "configs" / "experiment0.json")
