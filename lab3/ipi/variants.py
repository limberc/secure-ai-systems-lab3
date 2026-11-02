"""Lab 3, the two indirect-injection variants.

A variant turns the attacker's goal (harness.GOALS[scenario]) into the text that is
planted in the document or the email. The harness places it, runs the model, and decides
success in code -- you do not judge these by hand or by model.

`plain` is written: the goal, as-is. Liu et al. call this the naive attack, and it is the
baseline your two variants are measured against. Implement TWO published variants. The
natural source is Liu, Jia, Geng, Jia and Gong, "Formalizing and Benchmarking Prompt
Injection Attacks and Defenses", USENIX Security 2024, which defines

  escape characters   a newline or similar between the carrier text and the instruction
  context ignoring    "Ignore previous instructions ..." before the instruction
  fake completion     a fake answer to the original task ("Summary: done."), so the model
                      treats the task as finished and moves on to the instruction
  combined            all three together -- the paper's strongest attack

Greshake et al. (AISec 2023) is the other source: instructions dressed as content meant
for the assistant, or as a message from the system or the user.

Register each variant in VARIANTS under a short name; run.py ipi --variant <name> runs it.
"""


def plain(goal, scenario):
    """Liu et al.'s naive attack: the injected instruction, nothing else."""
    return goal


def variant_one(goal, scenario):
    raise NotImplementedError("Lab 3: implement your first indirect-injection variant")


def variant_two(goal, scenario):
    raise NotImplementedError("Lab 3: implement your second indirect-injection variant")


VARIANTS = {
    "plain": plain,
    "variant_one": variant_one,   # rename both, e.g. "fake_completion", "combined"
    "variant_two": variant_two,
}
