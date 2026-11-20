"""RuleParser: expected plans for varied commands. Steps are (skill, arm, obj[, other])."""
import pytest

from planner.language import DEFAULT_INSTRUCTION, RuleParser, plan_to_text

SKILLS = {"open_drawer", "pick", "place", "handoff", "pour"}
OBJS = {"plate", "mug", "bottle", "fork", "spoon", "drawer"}
DRAWER = ("open_drawer", "A", "drawer")
POUR_A = [("pick", "A", "bottle"), ("pick", "B", "mug"), ("pour", "A", "bottle", "B"),
          ("place", "A", "bottle"), ("place", "B", "mug")]

CASES = [
    (DEFAULT_INSTRUCTION, [
        DRAWER, ("pick", "A", "plate"), ("place", "A", "plate"), ("pick", "B", "spoon"), ("place", "B", "spoon"),
        ("pick", "B", "fork"), ("handoff", "B", "fork", "A"), ("place", "A", "fork"), ("pick", "B", "mug"),
        ("pick", "A", "bottle"), ("pour", "A", "bottle", "B"), ("place", "A", "bottle"), ("place", "B", "mug")]),
    ("open the drawer", [DRAWER]),
    ("Open the top drawer with arm B", [("open_drawer", "B", "drawer")]),
    ("pull the drawer open using the left arm", [DRAWER]),
    ("pick up the plate", [("pick", "A", "plate"), ("place", "A", "plate")]),
    ("grab the dish with arm A and put it on the table", [("pick", "A", "plate"), ("place", "A", "plate")]),
    ("pick up the plate and place it", [("pick", "A", "plate"), ("place", "A", "plate")]),
    ("Pick up the plate. Then place it.", [("pick", "A", "plate"), ("place", "A", "plate")]),
    ("place the plate", [("pick", "A", "plate"), ("place", "A", "plate")]),
    ("take the mug with the right arm", [("pick", "B", "mug"), ("place", "B", "mug")]),
    ("pick up the cup with arm B, then place it", [("pick", "B", "mug"), ("place", "B", "mug")]),
    ("lift the glass and set it down", [("pick", "B", "mug"), ("place", "B", "mug")]),
    ("put the mug on the table", [("pick", "B", "mug"), ("place", "B", "mug")]),
    ("set the plate and the mug on the table",
     [("pick", "A", "plate"), ("place", "A", "plate"), ("pick", "B", "mug"), ("place", "B", "mug")]),
    ("pick up the bottle; place it", [("pick", "A", "bottle"), ("place", "A", "bottle")]),
    ("grab the bottle and the mug",
     [("pick", "A", "bottle"), ("pick", "B", "mug"), ("place", "A", "bottle"), ("place", "B", "mug")]),
    ("pick up the fork", [DRAWER, ("pick", "B", "fork"), ("place", "B", "fork")]),
    ("take the spoon and place it on the table", [DRAWER, ("pick", "B", "spoon"), ("place", "B", "spoon")]),
    ("retrieve the spoon using arm B and lay it on the table",
     [DRAWER, ("pick", "B", "spoon"), ("place", "B", "spoon")]),
    ("open the drawer and take the spoon and put it on the table",
     [DRAWER, ("pick", "B", "spoon"), ("place", "B", "spoon")]),
    ("arrange the plate, then organize the fork",
     [DRAWER, ("pick", "A", "plate"), ("place", "A", "plate"), ("pick", "B", "fork"), ("place", "B", "fork")]),
    ("get the fork with arm B and hand it over to arm A",
     [DRAWER, ("pick", "B", "fork"), ("handoff", "B", "fork", "A"), ("place", "A", "fork")]),
    ("pick up the fork with arm B and hand it over to arm A, then place it",
     [DRAWER, ("pick", "B", "fork"), ("handoff", "B", "fork", "A"), ("place", "A", "fork")]),
    ("pass the spoon to the left arm",
     [DRAWER, ("pick", "B", "spoon"), ("handoff", "B", "spoon", "A"), ("place", "A", "spoon")]),
    ("give the fork to arm B",
     [DRAWER, ("pick", "A", "fork"), ("handoff", "A", "fork", "B"), ("place", "B", "fork")]),
    ("pour water into the mug", POUR_A),
    ("pour water into the mug with arm A", POUR_A),
    ("pick up the bottle with arm A, pick up the mug with arm B, pour water into the mug", POUR_A),
    ("pour water into the mug, then put the bottle back and place the mug on the table", POUR_A),
    ("pour water into the cup with arm B",
     [("pick", "B", "bottle"), ("pick", "A", "mug"), ("pour", "B", "bottle", "A"),
      ("place", "A", "mug"), ("place", "B", "bottle")]),
    ("", []),
    ("dance", []),
    ("place it", []),
]


def _short(plan):
    return [tuple(v for v in (s["skill"], s["arm"], s["obj"], s["other"]) if v) for s in plan]


@pytest.mark.parametrize("command,expected", CASES, ids=[c[:40] or "<empty>" for c, _ in CASES])
def test_plan(command, expected):
    assert _short(RuleParser().parse(command)) == expected


@pytest.mark.parametrize("command", [c for c, _ in CASES], ids=[c[:40] or "<empty>" for c, _ in CASES])
def test_plan_schema(command):
    for s in RuleParser().parse(command):
        assert set(s) == {"skill", "arm", "obj", "to", "other"}
        assert s["skill"] in SKILLS and s["arm"] in ("A", "B") and s["obj"] in OBJS
        if s["skill"] in ("handoff", "pour"):
            assert s["other"] in ("A", "B") and s["other"] != s["arm"]
        assert plan_to_text(s).startswith(f"arm {s['arm']} ")


def test_parse_is_deterministic():
    assert RuleParser().parse(DEFAULT_INSTRUCTION) == RuleParser().parse(DEFAULT_INSTRUCTION)


# Known defects, recorded so they cannot be forgotten. Remove the marker when fixed.
@pytest.mark.xfail(strict=True, reason="'hand X to arm Y' (no 'over') is not recognised as a hand-over; plan is empty")
def test_known_bug_hand_to():
    assert ("handoff", "B", "fork", "A") in _short(RuleParser().parse("hand the fork to arm A"))


@pytest.mark.xfail(strict=True, reason="unknown object yields a pick step with obj=None instead of being rejected")
def test_known_bug_unknown_object():
    assert all(s["obj"] is not None for s in RuleParser().parse("pick up the knife"))
