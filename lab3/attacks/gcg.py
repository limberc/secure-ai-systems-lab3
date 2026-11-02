"""Lab 3, family 3 of 3 (option B): simplified GCG -- Zou et al., "Universal and
Transferable Adversarial Attacks on Aligned Language Models", 2023.

Needs a CUDA GPU, the hf: backend (it reads gradients), and `pip install nanogcg`.
nanoGCG 0.3 works with transformers 4.x only -- installing it with pip downgrades
transformers to 4.47.1, which the rest of the harness also supports.
nanoGCG (Gray Swan AI) implements the paper's algorithm; this adapter runs it once per
behaviour to find a suffix, then sends goal + suffix k times like any template attack.

"Simplified" means a smaller search than the paper's: fewer steps, a narrower search,
optimising each behaviour separately rather than one universal suffix. Every such choice
is a deviation you state and justify. The constructor arguments below are the ones to
justify; the defaults are a starting point, not a recommendation.

Choose PAIR or GCG, not both. If you have no CUDA GPU, choose PAIR.
"""

from .base import TemplateAttack


class GCG(TemplateAttack):
    name = "gcg"
    family = "optimiser"
    citation = ("Zou et al., Universal and Transferable Adversarial Attacks on Aligned Language "
                "Models, 2023; implementation nanoGCG (Gray Swan AI)")

    def __init__(self, helper=None, num_steps=250, search_width=128, topk=64, seed=0):
        super().__init__(helper)
        self.config = dict(num_steps=num_steps, search_width=search_width, topk=topk, seed=seed)
        self.suffixes = {}

    def bind(self, target):
        from ..models import HFChat
        if not isinstance(target, HFChat) or target.device != "cuda":
            raise SystemExit("GCG needs the hf: backend on a CUDA GPU. Use PAIR otherwise.")
        import transformers
        if int(transformers.__version__.split(".")[0]) >= 5:
            raise SystemExit(f"nanoGCG 0.3 does not work with transformers {transformers.__version__}. "
                             "Run `pip install nanogcg`, which installs transformers 4.47.1.")
        self.target = target

    def suffix(self, behaviour):
        if behaviour["id"] not in self.suffixes:
            import nanogcg
            from nanogcg import GCGConfig
            cfg = GCGConfig(verbosity="WARNING", **self.config)
            # the suffix is optimised exactly where prompts() places it: after one space
            res = nanogcg.run(self.target.model, self.target.tok, behaviour["goal"] + " {optim_str}",
                              behaviour["target"], cfg)
            self.suffixes[behaviour["id"]] = (res.best_string, float(res.best_loss))
        return self.suffixes[behaviour["id"]]

    def prompts(self, behaviour, k):
        if "target" not in behaviour:
            raise SystemExit("GCG optimises against a target string, which benign twins do not have; "
                             "--wrap gcg is not supported. Say so in the report.")
        s, _ = self.suffix(behaviour)
        return [[{"role": "user", "content": f"{behaviour['goal']} {s}"}] for _ in range(k)]

    def describe(self):
        return {**super().describe(), "config": self.config,
                "suffixes": {b: {"suffix": s, "loss": l} for b, (s, l) in self.suffixes.items()}}
