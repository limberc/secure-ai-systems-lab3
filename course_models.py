"""The course model list. Lab 3 runs against these models and nothing else.

Every model a run touches -- target, transfer target, PAIR attacker, judge -- must be
named here, and `lab3.models.load` refuses anything that is not. It also refuses any
HTTP endpoint that is not on this machine unless it is listed in COURSE_ENDPOINTS.

Editing either list to point the harness at some other system is not a configuration
change. It is a breach of the Week 1 rules of engagement (ROE, section 2), judged on
what you did, not on what you intended.
"""

# alias -> (Hugging Face id, Ollama tag, role)
#
# role is advice, not enforcement: "target" models are the ones the brief asks you to
# attack; "judge" models are classifiers; "smoke" is for checking your setup and must
# not appear in reported results.
COURSE_MODELS = {
    "llama3.2-3b":    ("meta-llama/Llama-3.2-3B-Instruct",   "llama3.2:3b",     "target"),
    "llama3.1-8b":    ("meta-llama/Llama-3.1-8B-Instruct",   "llama3.1:8b",     "target"),
    "qwen2.5-3b":     ("Qwen/Qwen2.5-3B-Instruct",           "qwen2.5:3b",      "target"),
    "qwen2.5-7b":     ("Qwen/Qwen2.5-7B-Instruct",           "qwen2.5:7b",      "target"),
    "gemma2-2b":      ("google/gemma-2-2b-it",               "gemma2:2b",       "target"),
    "mistral-7b":     ("mistralai/Mistral-7B-Instruct-v0.3", "mistral:7b",      "target"),
    "llamaguard3-1b": ("meta-llama/Llama-Guard-3-1B",        "llama-guard3:1b", "judge"),
    "llamaguard3-8b": ("meta-llama/Llama-Guard-3-8B",        "llama-guard3:8b", "judge"),
    "qwen2.5-0.5b":   ("Qwen/Qwen2.5-0.5B-Instruct",         "qwen2.5:0.5b",    "smoke"),
}

# Model families, for the transfer requirement: the second target must come from a
# different family than the first.
FAMILY = {
    "llama3.2-3b": "llama", "llama3.1-8b": "llama",
    "qwen2.5-3b": "qwen", "qwen2.5-7b": "qwen", "qwen2.5-0.5b": "qwen",
    "gemma2-2b": "gemma", "mistral-7b": "mistral",
    "llamaguard3-1b": "llama", "llamaguard3-8b": "llama",
}

# Hosts other than this machine that the harness may call, as "host:port".
# Empty unless the instructor announces a shared course server on the LMS.
COURSE_ENDPOINTS = set()

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
