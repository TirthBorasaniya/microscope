"""IOI (indirect object identification) prompt generation.

Follows the standard template x name construction of Wang et al. 2022. Each prompt introduces
two names in its first clause; the *subject* (the repeated name, the giver) appears again before
a trailing " to". The *indirect object* is the other name and is the correct next-token
completion, e.g.::

    "When John and Mary went to the store, John gave a gift to"  ->  " Mary"

``make_ioi_prompts`` returns the dict described in the data contract.
"""

from __future__ import annotations

import random
from typing import Any

# Canonical IOI templates (Wang et al. 2022, BABA/ABBA families). Placeholders:
#   {A}, {B}  -> the two names introduced in the first clause
#   {S}       -> the subject (repeated name / giver)
#   {PLACE}, {OBJECT} -> filler nouns that add surface variety
# Every template ends with " to" so the model predicts the indirect-object name next.
IOI_TEMPLATES: list[str] = [
    "When {A} and {B} went to the {PLACE}, {S} gave a {OBJECT} to",
    "Then {A} and {B} went to the {PLACE}, and {S} gave a {OBJECT} to",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {S} decided to give it to",
    "After {A} and {B} left the {PLACE}, {S} handed the {OBJECT} to",
    "While {A} and {B} were at the {PLACE}, {S} passed the {OBJECT} to",
    "Then {A} and {B} had a long day at the {PLACE}, so {S} gave the {OBJECT} to",
    "When {A} and {B} arrived at the {PLACE}, {S} brought a {OBJECT} to",
    "The {PLACE} was where {A} and {B} met, and {S} offered the {OBJECT} to",
    "At the {PLACE}, {A} and {B} were talking, and {S} threw the {OBJECT} to",
    "Because {A} and {B} were near the {PLACE}, {S} carried the {OBJECT} to",
    "Then {A} and {B} stopped by the {PLACE}, where {S} sold a {OBJECT} to",
    "Once {A} and {B} reached the {PLACE}, {S} showed the {OBJECT} to",
    "When {A} and {B} finished at the {PLACE}, {S} lent the {OBJECT} to",
    "After visiting the {PLACE}, {A} and {B} talked, and {S} mailed a {OBJECT} to",
    "Inside the {PLACE}, {A} and {B} waited, and {S} read the {OBJECT} to",
]

# Names chosen to tokenize to a single token (with a leading space) under common tokenizers.
# Generation filters this pool to whatever is single-token for the supplied model.
IOI_NAMES: list[str] = [
    "Mary",
    "John",
    "Tom",
    "James",
    "Dan",
    "Paul",
    "Mark",
    "Mike",
    "Kevin",
    "Sarah",
    "Anna",
    "Laura",
]

IOI_PLACES: list[str] = [
    "store",
    "park",
    "school",
    "restaurant",
    "office",
    "garden",
    "hospital",
    "station",
]

IOI_OBJECTS: list[str] = [
    "drink",
    "gift",
    "book",
    "ring",
    "ball",
    "snack",
    "letter",
    "phone",
]

# Number of names required to assemble a single prompt (subject + indirect object).
_NAMES_PER_PROMPT = 2


def _is_single_token(model, text: str) -> bool:
    """True iff ``text`` tokenizes to exactly one token for ``model`` (no BOS prepended)."""
    try:
        tokens = model.to_tokens(text, prepend_bos=False)
    except Exception:
        return False
    return tokens.shape[-1] == 1


def make_ioi_prompts(model, n_prompts: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Generate ``n_prompts`` IOI prompts and the token ids / positions needed to score them.

    Parameters
    ----------
    model : HookedTransformer
        Used only for tokenization (``to_tokens`` / ``to_single_token``).
    n_prompts : int
        Number of prompts to generate (default mirrors the circuit-discovery config).
    seed : int
        Seed for reproducible template / name / filler sampling.

    Returns
    -------
    dict with keys ``prompts``, ``io_tokens``, ``subj_tokens``, ``positions`` as described in
    the data contract.
    """
    rng = random.Random(seed)

    # Keep only names that are a single token (with a leading space) for this tokenizer, so the
    # answer position is a clean one-token prediction target.
    single_token_names = [name for name in IOI_NAMES if _is_single_token(model, " " + name)]
    if len(single_token_names) < _NAMES_PER_PROMPT:
        raise ValueError(
            "Fewer than two single-token names available for this tokenizer; "
            "cannot build IOI prompts."
        )

    prompts: list[str] = []
    io_tokens: list[int] = []
    subj_tokens: list[int] = []
    positions: list[int] = []

    while len(prompts) < n_prompts:
        template = rng.choice(IOI_TEMPLATES)
        name_a, name_b = rng.sample(single_token_names, _NAMES_PER_PROMPT)
        subject = rng.choice([name_a, name_b])
        indirect_object = name_b if subject == name_a else name_a

        text = template.format(
            A=name_a,
            B=name_b,
            S=subject,
            PLACE=rng.choice(IOI_PLACES),
            OBJECT=rng.choice(IOI_OBJECTS),
        )

        tokens = model.to_tokens(text)  # (1, seq) with BOS prepended
        # The prediction is read at the final token of the prompt (the trailing " to").
        position = tokens.shape[-1] - 1

        prompts.append(text)
        io_tokens.append(model.to_single_token(" " + indirect_object))
        subj_tokens.append(model.to_single_token(" " + subject))
        positions.append(position)

    return {
        "prompts": prompts,
        "io_tokens": io_tokens,
        "subj_tokens": subj_tokens,
        "positions": positions,
    }
