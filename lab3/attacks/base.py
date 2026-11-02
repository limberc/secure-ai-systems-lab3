"""The attack interface, and the one attack that is already written: asking directly.

Two kinds of attack fit the harness.

TemplateAttack   turns a behaviour into k conversations up front -- one per attempt --
                 and, if it changes the form of the reply, decodes the reply back into
                 something a judge can read. Role-play, persuasion, encodings and GCG
                 (once its suffix is found) are template attacks.
IterativeAttack  talks to the target, reads the replies, and decides what to send next.
                 PAIR is iterative. Every target query it makes is an attempt.

A behaviour is a dict from data/harmful.jsonl: id, goal, target ("Sure, here is ..."),
behavior, category, source, split.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Attempt:
    messages: list            # the conversation sent to the target
    response: str             # what the target said, verbatim
    decoded: Optional[str] = None  # what the judge should read, if the attack changed the reply's form
    queries: int = 1          # target queries used up to and including this attempt
    meta: dict = field(default_factory=dict)


class TemplateAttack:
    name = "template"          # what the run is called; rename it to the technique you build
    family = ""                # persuasion_roleplay | encoding | optimiser | baseline
    citation = ""              # the paper you are reproducing -- required

    def __init__(self, helper=None):
        self.helper = helper   # optional course model, e.g. a paraphraser (--helper)

    def bind(self, target):
        """Called once with the loaded target before any prompts are built. Only white-box
        attacks (GCG) need it."""

    def prompts(self, behaviour, k):
        """k conversations, one per attempt. Each is a list of {"role", "content"} dicts.
        Returning the same conversation k times is fine: attempts then differ only by
        sampling seed."""
        raise NotImplementedError

    def decode(self, response, messages=None, behaviour=None, attempt=None):
        """Map the target's reply back to plain text for the judge. Identity by default.

        `messages` is the conversation that produced the reply, `behaviour` the item and
        `attempt` its index -- enough to decode attempt j with scheme j. If you fix
        decode() after a run, `run.py redecode <run>` re-applies it to the stored replies;
        you do not need to generate again."""
        return response

    def describe(self):
        return {"name": self.name, "family": self.family, "citation": self.citation,
                "class": type(self).__name__}


class IterativeAttack:
    name = "iterative"
    family = "optimiser"
    citation = ""

    def run(self, behaviour, target, gen):
        """Attack one behaviour; return the list of Attempts in the order they were made."""
        raise NotImplementedError

    def describe(self):
        return {"name": self.name, "family": self.family, "citation": self.citation,
                "class": type(self).__name__}


class DirectRequest(TemplateAttack):
    """The undefended baseline: the behaviour's goal, sent as the user's message.

    Every other attack is a difference from this number. Run it on the same split, the
    same target and the same k before you run anything else.
    """

    name = "direct"
    family = "baseline"
    citation = "No attack -- the request as written in JBB-Behaviors"

    def prompts(self, behaviour, k):
        return [[{"role": "user", "content": behaviour["goal"]}] for _ in range(k)]
