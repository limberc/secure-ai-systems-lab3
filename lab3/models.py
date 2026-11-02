"""Model backends: one `chat()` interface over three ways of running a course model.

A model spec is  backend:alias[@base_url]

    hf:qwen2.5-3b                         Hugging Face transformers, in this process
    openai:qwen2.5-3b                     an OpenAI-compatible server on this machine;
                                          defaults to Ollama at http://localhost:11434/v1
    openai:qwen2.5-7b@http://localhost:8000/v1    e.g. a local `vllm serve`
    mock                                  not a model -- a scripted stand-in for tests

`alias` must be a key of course_models.COURSE_MODELS, and an openai base_url must be on
this machine (or in COURSE_ENDPOINTS). Nothing else loads.
"""

import json
import platform
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from course_models import COURSE_ENDPOINTS, COURSE_MODELS, LOCAL_HOSTS


@dataclass
class Gen:
    """Generation settings. Recorded in every run header."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 0

    def asdict(self):
        return asdict(self)


TEMPLATE_DATE = "02 Nov 2026"


class OutOfScope(Exception):
    """Raised when a spec names a model or endpoint the course has not authorised."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OutOfScope(f"{req.full_url} redirected to {newurl}; the harness does not follow redirects")


# No proxy (HTTP_PROXY would otherwise carry every prompt off this machine) and no
# redirects (a local server could otherwise hand the request to somewhere else).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def parse_spec(spec):
    if spec == "mock":
        return "mock", "mock", None
    backend, _, rest = spec.partition(":")
    alias, _, base_url = rest.partition("@")
    if backend not in ("hf", "openai"):
        raise ValueError(f"unknown backend {backend!r} in {spec!r}; use hf:, openai: or mock")
    if alias not in COURSE_MODELS:
        raise OutOfScope(f"{alias!r} is not on the course model list (course_models.py). "
                         "Lab 3 runs against course models only.")
    return backend, alias, base_url or None


def check_endpoint(base_url):
    u = urllib.parse.urlparse(base_url)
    host, port = u.hostname or "", u.port
    if host in LOCAL_HOSTS or f"{host}:{port}" in COURSE_ENDPOINTS:
        return
    raise OutOfScope(f"{base_url} is not on this machine and not a course endpoint. "
                     "Red-team traffic to a hosted service reaches its operator; "
                     "Lab 3 targets run locally.")


def load(spec, **kw):
    backend, alias, base_url = parse_spec(spec)
    if backend == "mock":
        return MockChat()
    if backend == "hf":
        return HFChat(alias, **kw)
    return OpenAIChat(alias, base_url or "http://localhost:11434/v1", **kw)


def fold_system(messages):
    """Merge a leading system message into the first user turn, for chat templates
    (Gemma 2, some Mistral versions) that reject the system role."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_msg, rest = messages[0]["content"], list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{sys_msg}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": sys_msg})
    return rest


def hub_revision(hf_id):
    """The commit hash of the cached snapshot the weights were loaded from."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(hf_id, "config.json")
        parts = str(path).split("/")
        return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    except Exception:
        return None


def _mps_bf16(torch):
    try:
        return bool((torch.ones(1, dtype=torch.bfloat16, device="mps") * 2).item() == 2)
    except Exception:
        return False


class HFChat:
    """A course model loaded with transformers. Batched, left-padded, seeded per batch."""

    def __init__(self, alias, batch_size=8, device=None, load_in_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.alias, self.hf_id = alias, COURSE_MODELS[alias][0]
        self.batch_size = batch_size
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
        if alias.startswith("gemma") and dtype == torch.float16:
            # Gemma 2's activations overflow float16: bfloat16 where the Mac supports it,
            # float32 otherwise (a T4 has no bfloat16)
            dtype = torch.bfloat16 if device == "mps" and _mps_bf16(torch) else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        import transformers
        from packaging.version import Version
        # transformers 4.56 renamed torch_dtype to dtype; v5 dropped the old name
        dkey = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        kwargs: dict = {dkey: dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                               bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs)
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.device, self.dtype, self.load_in_4bit = device, str(dtype), load_in_4bit
        self._torch = torch

    def render(self, messages):
        # date_string pins the date the Llama 3.x templates write into the system turn,
        # which otherwise changes with the day you run; other templates ignore it
        kw = dict(tokenize=False, add_generation_prompt=True, date_string=TEMPLATE_DATE)
        try:
            return self.tok.apply_chat_template(messages, **kw)
        except Exception:
            return self.tok.apply_chat_template(fold_system(messages), **kw)

    def chat(self, conversations, gen):
        torch = self._torch
        texts = [self.render(m) for m in conversations]
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            torch.manual_seed(gen.seed + i)
            kwargs = dict(max_new_tokens=gen.max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if gen.temperature > 0:
                kwargs.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p)
            else:
                kwargs.update(do_sample=False, temperature=None, top_p=None)
            with torch.no_grad():
                ids = self.model.generate(**enc, **kwargs)
            new = ids[:, enc["input_ids"].shape[1]:]
            # no clean-up: it strips spaces before punctuation in BPE models, and the
            # record should hold the reply exactly as the model produced it
            out += [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)]
        return out

    def info(self):
        import torch
        import transformers
        return {"backend": "hf", "alias": self.alias, "hf_id": self.hf_id,
                "revision": hub_revision(self.hf_id) or getattr(self.model.config, "_commit_hash", None),
                "dtype": self.dtype, "device": self.device, "load_in_4bit": self.load_in_4bit,
                "batch_size": self.batch_size, "template_date": TEMPLATE_DATE,
                "transformers": transformers.__version__, "torch": torch.__version__}


class OpenAIChat:
    """A course model behind an OpenAI-compatible endpoint on this machine (Ollama, vLLM,
    LM Studio). Ollama is addressed by its tag; anything else by the Hugging Face id."""

    def __init__(self, alias, base_url, workers=4, timeout=600):
        check_endpoint(base_url)
        self.alias, self.base_url = alias, base_url.rstrip("/")
        hf_id, ollama_tag, _ = COURSE_MODELS[alias]
        port = urllib.parse.urlparse(base_url).port
        self.served = ollama_tag if port == 11434 else hf_id
        self.workers, self.timeout = workers, timeout
        self._seen_model = None

    def _one(self, messages, gen, seed):
        body = {"model": self.served, "messages": messages, "temperature": gen.temperature,
                "top_p": gen.top_p, "max_tokens": gen.max_new_tokens, "seed": seed}
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer local"})
        with _OPENER.open(req, timeout=self.timeout) as r:
            data = json.load(r)
        self._seen_model = data.get("model", self._seen_model)
        return (data["choices"][0]["message"].get("content") or "").strip()

    def chat(self, conversations, gen):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda a: self._one(a[1], gen, gen.seed + a[0]),
                                 enumerate(conversations)))

    def info(self):
        return {"backend": "openai", "alias": self.alias, "hf_id": COURSE_MODELS[self.alias][0],
                "served_as": self.served, "reported_model": self._seen_model,
                "base_url": self.base_url}


class MockChat:
    """Not a model. A scripted stand-in so the harness and tests run without weights.

    It refuses any request that plainly contains a word from REFUSE, complies with
    everything else, and -- deliberately naive -- carries out any instruction it finds
    that mentions the lab's canary domain. Numbers measured on it mean nothing.
    """

    REFUSE = ("defamatory", "malware", "hack", "steal", "scam", "fake", "illegal", "phishing")
    CANARY = "canary.lab3.example"

    alias = "mock"

    def _reply(self, messages):
        text = "\n".join(m["content"] for m in messages)
        last = messages[-1]["content"].lower()
        if self.CANARY in text:
            if "ACTION send_email" in text:
                return ('ACTION send_email {"to": "collector@canary.lab3.example", '
                        '"subject": "fwd", "body": "forwarded"}')
            return f"Summary. Please re-verify at http://{self.CANARY}/verify"
        if "ACTION send_email" in text:
            return "FINAL: You have three unread emails."
        if any(w in last for w in self.REFUSE):
            return "I'm sorry, but I can't help with that."
        return "Sure, here is a response. " + re.sub(r"\s+", " ", last)[:80]

    def chat(self, conversations, gen):
        return [self._reply(m) for m in conversations]

    def info(self):
        return {"backend": "mock", "alias": "mock", "python": platform.python_version()}
