"""Lab 3, family 1 of 3: one persuasion or role-play technique, reproduced faithfully.

Pick ONE published technique from the competing-objectives side of Week 8's taxonomy and
implement it here. Candidates, all in the Week 8 deck:

  * role-play or persona framing -- Wei, Haghtalab and Steinhardt, "Jailbroken", NeurIPS 2023
  * prefix injection or refusal suppression -- the same paper, which gives the exact wording
  * persuasive paraphrase (PAP) -- Zeng et al., "How Johnny Can Persuade LLMs to Jailbreak
    Them", ACL 2024. The authors used a fine-tuned paraphraser; prompting a course model
    instead (pass --helper) is a deviation you must state.

"Faithful" is the word the marks hang on. Use the published wording or procedure where the
paper gives it, and write down every place you depart from it and why. A faithful
reproduction that succeeds less often scores better than an embellished one that succeeds
more, because the point is to measure what the published technique does.

Tune on --split dev. Do not change this file after your first --split test run without
saying so in the report.
"""

from .base import TemplateAttack


class PersuasionRoleplay(TemplateAttack):
    name = "persuasion_roleplay"   # rename to the technique, e.g. "refusal_suppression"
    family = "persuasion_roleplay"
    citation = ""                  # required: author, title, venue, year

    def prompts(self, behaviour, k):
        # behaviour["goal"] is the request. Return k conversations.
        # If your technique has several published templates, attempt j may use template j.
        raise NotImplementedError("Lab 3: implement your persuasion or role-play family here")
