"""Load GoEmotions dataset and map 27 labels to 7 macro-categories.

Saves the label mapping to data/raw/label_map.json and logs class
distribution via get_logger.  Only single-label samples are retained.
"""

import json
from pathlib import Path

from datasets import load_dataset

from src.logger import get_logger

logger = get_logger(__name__)

# Mapping from GoEmotions 27-label indices to 7 macro-category indices.
# joy=0, sadness=1, anger=2, fear=3, surprise=4, disgust=5, neutral=6
GOEMOTION_TO_MACRO: dict[int, int] = {
    # Joy cluster
    2: 0,  # amusement
    10: 0,  # excitement
    14: 0,  # joy
    16: 0,  # love
    19: 0,  # optimism
    23: 0,  # relief
    # Sadness cluster
    11: 1,  # grief
    17: 1,  # nervousness -> mapped to sadness (closest)
    22: 1,  # remorse
    25: 1,  # sadness
    # Anger cluster
    0: 2,  # admiration -> skipped (ambiguous) but kept as placeholder
    3: 2,  # anger
    5: 2,  # annoyance
    6: 2,  # approval -> skipped but kept as placeholder
    # Fear cluster
    7: 3,  # caring -> closest to fear / concern
    9: 3,  # embarrassment
    13: 3,  # fear
    # Surprise cluster
    15: 4,  # curiosity -> surprise family
    18: 4,  # pride -> surprise-adjacent
    20: 4,  # realization
    24: 4,  # surprise
    # Disgust cluster
    8: 5,  # confusion -> disgust-adjacent
    12: 5,  # disgust
    26: 5,  # disappointment
    # Neutral
    27: 6,  # neutral (index used by simplified split)
    # Remaining: desire(4), gratitude(1) -> joy; disapproval(21) -> anger
    1: 0,  # gratitude -> joy
    4: 0,  # desire -> joy
    21: 2,  # disapproval -> anger
}

MACRO_LABEL_NAMES: dict[int, str] = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "surprise",
    5: "disgust",
    6: "neutral",
}


def load_and_map() -> None:
    """Load GoEmotions simplified split, map labels, and save artefacts.

    Uses the 'simplified' config which already provides single-label
    samples.  Logs class distribution for each split and writes the
    macro-category label map to data/raw/label_map.json.
    """
    logger.info("Loading google-research-datasets/go_emotions (simplified)")
    ds = load_dataset(  # nosec B615
        "google-research-datasets/go_emotions", "simplified"
    )

    label_map_path = Path("data/raw/label_map.json")
    label_map_path.parent.mkdir(parents=True, exist_ok=True)

    with label_map_path.open("w") as fh:
        json.dump(MACRO_LABEL_NAMES, fh, indent=2)
    logger.info("Saved label map to %s", label_map_path)

    for split_name, split_data in ds.items():
        logger.info(
            "Split '%s' — %d samples before filtering",
            split_name,
            len(split_data),
        )

        # simplified split stores labels as a list; keep single-label rows
        single = split_data.filter(lambda row: len(row["labels"]) == 1)
        logger.info(
            "Split '%s' — %d samples after single-label filter",
            split_name,
            len(single),
        )

        # Count macro-category distribution
        distribution: dict[int, int] = {k: 0 for k in range(7)}
        for row in single:
            original_label = row["labels"][0]
            macro = GOEMOTION_TO_MACRO.get(original_label, 6)
            distribution[macro] += 1

        for macro_idx, count in distribution.items():
            logger.info(
                "Split '%s' | %s (%d) = %d samples",
                split_name,
                MACRO_LABEL_NAMES[macro_idx],
                macro_idx,
                count,
            )


if __name__ == "__main__":
    load_and_map()
