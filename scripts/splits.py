"""Seed splits. Single source of truth; see results/SPLITS.md for the rationale."""

TRAIN = "100-399"      # demonstration seeds (data/demos)
DEV = "400-499"        # model selection, tuning, ablations
TEST = "1000-1099"     # final numbers only; never tuned on
LEGACY = "0-19"        # seeds the scripted skills were developed on; only for reproducing old tables

SPLITS = dict(train=TRAIN, dev=DEV, test=TEST, legacy=LEGACY)


def seeds(name: str) -> list[int]:
    a, b = SPLITS[name].split("-")
    return list(range(int(a), int(b) + 1))


def split_of(seed: int) -> str | None:
    for name in SPLITS:
        a, b = (int(x) for x in SPLITS[name].split("-"))
        if a <= seed <= b:
            return name
    return None
