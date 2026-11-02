"""Attacks, by the name run.py knows them."""

from .base import Attempt, DirectRequest, IterativeAttack, TemplateAttack
from .encoding import Encoding
from .gcg import GCG
from .pair import PAIR
from .persuasion_roleplay import PersuasionRoleplay

TEMPLATE_ATTACKS = {
    "direct": DirectRequest,
    "persuasion_roleplay": PersuasionRoleplay,
    "encoding": Encoding,
    "gcg": GCG,
}

__all__ = ["Attempt", "DirectRequest", "Encoding", "GCG", "IterativeAttack", "PAIR",
           "PersuasionRoleplay", "TEMPLATE_ATTACKS", "TemplateAttack"]
