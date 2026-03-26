"""Pandera schema for validating the processed emotion dataset.

EMOTION_SCHEMA must be called on any DataFrame before a train/val/test split
is performed.  Never validate after splitting — that would miss cross-split
leakage issues.
"""

from pandera import Check, Column, DataFrameSchema

EMOTION_SCHEMA = DataFrameSchema(
    {
        "text": Column(str, nullable=False),
        "label": Column(
            int,
            Check.isin([0, 1, 2, 3, 4, 5, 6]),
            nullable=False,
        ),
    },
    strict=True,
)
"""DataFrameSchema that enforces the processed GoEmotions contract.

Expected columns
----------------
text  : str, non-null — raw Reddit comment text
label : int in {0,1,2,3,4,5,6}, non-null — macro-emotion index

strict=True rejects any DataFrame that contains extra columns beyond
'text' and 'label'.
"""
