"""Lab 3, family 2 of 3: one encoding attack, reproduced faithfully.

The mismatched-generalisation side of Week 8's taxonomy: the model can read a form that
its safety training rarely saw. Candidates:

  * Base64, input only, or input and output -- Wei, Haghtalab and Steinhardt, NeurIPS 2023
  * ROT13, leetspeak, or another character-level transform from the same paper
  * translation into a low-resource language -- Yong, Menghini and Bach, 2023
    (the translation step must use a course model or an offline tool, never a hosted API)

Two things the harness cannot do for you.

1. decode(). If your attack asks for an encoded reply, the judge must read the decoded
   text. A Base64 reply judged as Base64 counts as "no refusal phrase", whatever it says.
2. Capability. Wei et al. found Base64 worked on GPT-4 and not on weaker models, which
   could not read it. A 3B model may answer your encoded request with nonsense. Run the
   same wrapper on the benign twins (`run.py benign --wrap encoding`): if the model cannot
   answer a harmless request in this form, its "successes" on harmful ones are not
   jailbreaks, and the rubric judge's BENIGN label is how you will see it.
"""

from .base import TemplateAttack


class Encoding(TemplateAttack):
    name = "encoding"      # rename to the technique, e.g. "base64_input_only"
    family = "encoding"
    citation = ""          # required

    def prompts(self, behaviour, k):
        raise NotImplementedError("Lab 3: implement your encoding family here")

    def decode(self, response, messages=None, behaviour=None, attempt=None):
        # Return the plain-text reading of the target's reply. Identity is correct only
        # if your attack does not ask for an encoded answer. `messages` is what was sent,
        # so a decoder can tell which scheme attempt j used.
        return response
