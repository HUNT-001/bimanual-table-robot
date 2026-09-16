"""Natural-language instruction -> symbolic bimanual plan.

Two interchangeable back-ends:
  * RuleParser  - deterministic, zero-dependency grammar (default; used for evaluation).
  * LLMParser   - optional small instruction-tuned LLM (e.g. Qwen2.5-0.5B-Instruct exported to
                  OpenVINO IR with optimum-intel) that emits the same JSON plan; falls back to rules.

Plan step schema: {"skill": str, "arm": "A"|"B", "obj": str|None, "to": str|None, "other": str|None}
"""
from __future__ import annotations

import json
import re

OBJ_SYNONYMS = {
    "plate": ["plate", "dish"],
    "mug": ["mug", "cup", "glass"],
    "bottle": ["bottle", "jug", "water"],
    "fork": ["fork", "forks"],
    "spoon": ["spoon", "spoons"],
    "drawer": ["drawer"],
}
ARM_RE = re.compile(r"\b(?:with|using|by)\s+(?:the\s+)?(?:arm\s*([ab])|([ab])\s*arm|(left|right)\s+arm)\b", re.I)
RECV_RE = re.compile(r"\bto\s+(?:the\s+)?(?:arm\s*([ab])\b|(left|right)\s+arm)", re.I)
DEFAULT_ARM = {"plate": "A", "bottle": "A", "mug": "B", "spoon": "B", "fork": "B", "drawer": "A"}

DEFAULT_INSTRUCTION = (
    "Open the top drawer with arm A, pick up the plate with arm A and place it on the table, "
    "take the spoon with arm B and place it on the table, pick up the fork with arm B and hand it over to arm A, "
    "then place the fork, pick up the mug with arm B, pour water into the mug with arm A, "
    "then put the bottle back and place the mug on the table."
)


def _find_obj(text: str):
    t = text.lower()
    hits = []
    for obj, syns in OBJ_SYNONYMS.items():
        for s in syns:
            m = re.search(rf"\b{s}\b", t)
            if m:
                hits.append((m.start(), obj))
    hits.sort()
    return [o for _, o in hits]


def _arm(text: str):
    m = ARM_RE.search(text)
    if not m:
        return None
    a, b, side = m.groups()
    if a or b:
        return (a or b).upper()
    return "A" if side.lower() == "left" else "B"


class RuleParser:
    """Clause-level grammar. Keeps a context of 'what each arm is holding' across clauses,
    so pronouns ('place it') and implicit arguments resolve correctly."""

    def parse(self, instruction: str) -> list[dict]:
        verbs = r"(?:pick|place|put|set|take|hand|give|pass|pour|open|grab|get|lay|lift|retrieve|arrange|organi[sz]e)"
        clauses = re.split(rf",|;|\bthen\b|\.\s|\band\b(?=\s+(?:then\s+)?{verbs}\b)", instruction, flags=re.I)
        plan, holding, last_obj = [], {"A": None, "B": None}, None
        for c in clauses:
            c = c.strip()
            if not c:
                continue
            lc = c.lower()
            objs = _find_obj(lc)
            arm = _arm(c)
            # drawer
            if "drawer" in lc and re.search(r"\bopen|pull\b", lc):
                plan.append(dict(skill="open_drawer", arm=arm or "A", obj="drawer"))
                continue
            # hand-over
            if re.search(r"hand(\s*it)?\s*(over|off)|pass|give", lc):
                obj = next((o for o in objs if o != "drawer"), None) or last_obj
                rm = RECV_RE.search(c)
                recv = None
                if rm:
                    a, side = rm.groups()
                    recv = a.upper() if a else ("A" if side.lower() == "left" else "B")
                giver = next((a for a, h in holding.items() if h == obj), None)
                if giver is None:
                    giver = arm if arm and arm != recv else ("B" if recv == "A" else "A")
                recv = recv or ("A" if giver == "B" else "B")
                if holding[giver] != obj:
                    plan.append(dict(skill="pick", arm=giver, obj=obj))
                plan.append(dict(skill="handoff", arm=giver, other=recv, obj=obj))
                holding[giver], holding[recv], last_obj = None, obj, obj
                continue
            # pour
            if re.search(r"\bpour", lc):
                into = "mug"
                pourer = arm or next((a for a, h in holding.items() if h == "bottle"), None) or "A"
                holder = "B" if pourer == "A" else "A"
                if holding[pourer] != "bottle":
                    plan.append(dict(skill="pick", arm=pourer, obj="bottle"))
                    holding[pourer] = "bottle"
                if holding[holder] != into:
                    plan.append(dict(skill="pick", arm=holder, obj=into))
                    holding[holder] = into
                plan.append(dict(skill="pour", arm=pourer, other=holder, obj="bottle", to=into))
                last_obj = into
                continue
            # pick (+ optional place in same clause)
            is_pick = re.search(r"\b(pick|grab|take|get|retrieve|lift)\b", lc)
            is_place = re.search(r"\b(place|put|set|lay|organi[sz]e|arrange)\b", lc)
            if is_pick:
                for obj in [o for o in objs if o != "drawer"] or [last_obj]:
                    a = arm or DEFAULT_ARM.get(obj, "A")
                    plan.append(dict(skill="pick", arm=a, obj=obj))
                    holding[a], last_obj = obj, obj
                    if is_place:
                        plan.append(dict(skill="place", arm=a, obj=obj, to=obj))
                        holding[a] = None
                continue
            if is_place:
                targets = [o for o in objs if o not in ("drawer",)]
                if re.search(r"\b(it|them)\b", lc) or not targets:
                    targets = [last_obj] if last_obj else []
                for obj in targets:
                    a = next((x for x, h in holding.items() if h == obj), None)
                    if a is None:
                        a = arm or DEFAULT_ARM.get(obj, "A")
                        plan.append(dict(skill="pick", arm=a, obj=obj))
                    plan.append(dict(skill="place", arm=a, obj=obj, to=obj))
                    holding[a] = None
                    last_obj = obj
        # anything still in hand gets put down at the end
        for a, h in holding.items():
            if h is not None:
                plan.append(dict(skill="place", arm=a, obj=h, to=h))
        # dependency: cutlery lives in the drawer -> ensure it is opened first
        needs_drawer = any(s.get("obj") in ("fork", "spoon") for s in plan)
        if needs_drawer and not any(s["skill"] == "open_drawer" for s in plan):
            plan.insert(0, dict(skill="open_drawer", arm="A", obj="drawer"))
        return [{k: s.get(k) for k in ("skill", "arm", "obj", "to", "other")} for s in plan]


class LLMParser:
    """Optional: OpenVINO GenAI LLM planner. Falls back to RuleParser on any failure."""

    SYSTEM = ("You convert robot table-setting commands into a JSON list of steps. Skills: open_drawer, pick, "
              "place, handoff, pour. Arms: A (left), B (right). Objects: plate, mug, bottle, fork, spoon. "
              "Each step: {\"skill\",\"arm\",\"obj\",\"to\",\"other\"}. Output JSON only.")

    def __init__(self, model_dir: str, device: str = "CPU"):
        self.fallback = RuleParser()
        try:
            import openvino_genai as ov_genai  # noqa
            self.pipe = ov_genai.LLMPipeline(model_dir, device)
        except Exception as exc:  # pragma: no cover
            print(f"[LLMParser] unavailable ({exc}); using RuleParser")
            self.pipe = None

    def parse(self, instruction: str):
        if self.pipe is None:
            return self.fallback.parse(instruction)
        try:
            out = self.pipe.generate(f"{self.SYSTEM}\nCommand: {instruction}\nJSON:", max_new_tokens=400)
            plan = json.loads(out[out.index("["): out.rindex("]") + 1])
            assert all(p["skill"] in ("open_drawer", "pick", "place", "handoff", "pour") for p in plan)
            return plan
        except Exception:
            return self.fallback.parse(instruction)


def plan_to_text(step: dict) -> str:
    """Canonical sub-instruction string (used to condition the learned policy)."""
    s = step["skill"]
    if s == "open_drawer":
        return f"arm {step['arm']} open the drawer"
    if s == "pick":
        return f"arm {step['arm']} pick up the {step['obj']}"
    if s == "place":
        return f"arm {step['arm']} place the {step['obj']} on its spot"
    if s == "handoff":
        return f"arm {step['arm']} hand the {step['obj']} to arm {step['other']}"
    if s == "pour":
        return f"arm {step['arm']} pour from the bottle into the mug held by arm {step['other']}"
    return s


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or DEFAULT_INSTRUCTION
    for i, st in enumerate(RuleParser().parse(text)):
        print(i, plan_to_text(st))
