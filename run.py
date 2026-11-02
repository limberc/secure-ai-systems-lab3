#!/usr/bin/env python3
"""Lab 3 runner. Every command writes one run directory under out/runs/ (see lab3/records.py).

  jailbreak  a template attack on the harmful set        run.py jailbreak --attack direct --target hf:qwen2.5-3b --split dev
  pair       PAIR on the harmful set                     run.py pair --target hf:qwen2.5-3b --attacker hf:mistral-7b --split dev
  benign     the benign twins or XSTest, optionally      run.py benign --set twins --target hf:qwen2.5-3b --split dev
             through an attack's wrapper (--wrap)
  transfer   replay one run's successes on a second      run.py transfer --source <run> --judge rubric-qwen2.5-7b --target hf:llama3.2-3b
             target
  judge      add a judge's verdicts to a run             run.py judge <run> --judge rubric:hf:qwen2.5-7b
  redecode   re-apply a fixed decode() to stored replies  run.py redecode <run>
  ipi        an indirect-injection variant               run.py ipi --scenario doc --variant plain --target hf:qwen2.5-3b --split dev

Generation, judging and scoring are separate steps, so the target and the judge never need
to be in memory together. Then: label.py for your hand labels, score.py for the tables.
"""

import argparse
import json
import sys
import time
from dataclasses import replace

from lab3 import data, judges, models, records
from lab3.attacks import PAIR, TEMPLATE_ATTACKS, DirectRequest
from lab3.ipi import harness
from lab3.ipi.variants import VARIANTS


def model_kw(spec, a):
    return {"batch_size": a.batch_size, "load_in_4bit": a.load_in_4bit} if spec.startswith("hf:") else {}


def load(spec, a, cache):
    """Load each distinct spec once, so attacker == scorer does not mean two copies."""
    if spec not in cache:
        cache[spec] = models.load(spec, **model_kw(spec, a))
    return cache[spec]


def gen_of(a):
    return models.Gen(temperature=a.temperature, top_p=a.top_p,
                      max_new_tokens=a.max_new_tokens, seed=a.seed)


def build_prompts(attack, items, k):
    """All prompts up front, before a run directory exists: an attack that fails here
    leaves nothing behind in out/."""
    return {it["id"]: attack.prompts(it, k) for it in items}


def run_template(attack, target, items, k, gen, run, convs=None):
    convs = convs or build_prompts(attack, items, k)
    for j in range(k):
        batch = [convs[it["id"]][j] for it in items]
        outs = target.chat(batch, replace(gen, seed=gen.seed + 1000 * j))
        for it, conv, out in zip(items, batch, outs):
            dec = attack.decode(out, messages=conv, behaviour=it, attempt=j)
            run.add({"key": f"{it['id']}:{j}", "behaviour_id": it["id"],
                     "category": it.get("category"), "goal": it["goal"], "attempt": j,
                     "messages": conv, "response": out,
                     "decoded": dec if dec != out else None, "queries": j + 1})
        print(f"  attempt {j + 1}/{k}: {len(items)} responses", file=sys.stderr)


def cmd_jailbreak(a):
    cache = {}
    helper = load(a.helper, a, cache) if a.helper else None
    attack = TEMPLATE_ATTACKS[a.attack](helper=helper)
    target = load(a.target, a, cache)
    attack.bind(target)
    items = data.load("harmful", a.split, a.limit)
    convs = build_prompts(attack, items, a.k)
    gen = gen_of(a)
    run = records.Run("jailbreak", attack.name, target.alias, a.split, {
        "attack": attack.describe(), "target_model": target.info(), "helper_spec": a.helper,
        "helper_model": helper.info() if helper else None, "gen": gen.asdict(), "k": a.k,
        "behaviour_ids": [it["id"] for it in items]})
    run_template(attack, target, items, a.k, gen, run, convs)
    run.header["attack"] = attack.describe()      # GCG adds its suffixes here
    print(run.close())


def cmd_pair(a):
    cache = {}
    target = load(a.target, a, cache)
    attacker = load(a.attacker, a, cache)
    scorer = load(a.scorer or a.attacker, a, cache)
    if attacker is target or scorer is target:
        print("warning: the attacker or the in-loop judge is the target model. Chao et al. kept "
              "them separate; state this as a deviation.", file=sys.stderr)
    attack = PAIR(attacker, scorer, streams=a.streams, iterations=a.iterations)
    items = data.load("harmful", a.split, a.limit)
    gen = gen_of(a)
    run = records.Run("pair", attack.name, target.alias, a.split, {
        "attack": attack.describe(), "target_model": target.info(), "gen": gen.asdict(),
        "k": a.streams * a.iterations, "behaviour_ids": [it["id"] for it in items]})
    for n, it in enumerate(items, 1):
        atts = attack.run(it, target, gen)
        for idx, at in enumerate(atts):
            run.add({"key": f"{it['id']}:{idx}", "behaviour_id": it["id"],
                     "category": it.get("category"), "goal": it["goal"], "attempt": idx,
                     "messages": at.messages, "response": at.response, "decoded": at.decoded,
                     "queries": at.queries, "meta": at.meta})
        best = max((at.meta.get("inloop_score", 0) for at in atts), default=0)
        print(f"  {n}/{len(items)} {it['id']}: {len(atts)} queries, best in-loop score {best}",
              file=sys.stderr)
    print(run.close())


def cmd_benign(a):
    cache = {}
    helper = load(a.helper, a, cache) if a.helper else None
    wrap = TEMPLATE_ATTACKS[a.wrap](helper=helper) if a.wrap else DirectRequest()
    target = load(a.target, a, cache)
    wrap.bind(target)
    if a.set == "xstest":
        if a.split != "test":
            print("note: XSTest has no dev split; using all 250 safe prompts.", file=sys.stderr)
        items, split = data.load("xstest", None, a.limit), "test"
    else:
        items, split = data.load("twins", a.split, a.limit), a.split
    convs = build_prompts(wrap, items, a.k)
    gen = gen_of(a)
    name = a.set + (f"+{wrap.name}" if a.wrap else "")
    run = records.Run("benign", name, target.alias, split, {
        "set": a.set, "wrap": wrap.describe() if a.wrap else None, "helper_spec": a.helper,
        "target_model": target.info(),
        "gen": gen.asdict(), "k": a.k, "behaviour_ids": [it["id"] for it in items]})
    run_template(wrap, target, items, a.k, gen, run, convs)
    print(run.close())


def cmd_transfer(a):
    src = records.resolve(a.source)
    sh, srecs = records.read(src)
    verdicts = records.read_judgements(src).get(a.judge)
    if verdicts is None:
        raise SystemExit(f"{a.judge}: no such judgement in {src.name}. Run `run.py judge` on it first.")
    if sh["kind"] not in ("jailbreak", "pair"):
        raise SystemExit("transfer replays a jailbreak or pair run")
    cache = {}
    decoder = source_attack(sh, a, cache) if sh["kind"] == "jailbreak" else DirectRequest()
    first = {}
    for r in sorted(srecs, key=lambda r: r["attempt"]):
        if verdicts.get(r["key"], {}).get("harmful") and r["behaviour_id"] not in first:
            first[r["behaviour_id"]] = r
    target = load(a.target, a, cache)
    if target.alias == sh["target"]:
        raise SystemExit("transfer needs a different target from the source run")
    from course_models import FAMILY
    if FAMILY.get(target.alias) == FAMILY.get(sh["target"]):
        print(f"warning: {target.alias} and {sh['target']} are the same family; the brief asks for a different one.",
              file=sys.stderr)
    gen = gen_of(a)
    src_items = list(first.values())
    items = {it["id"]: it for it in data.load("harmful")}
    run = records.Run("transfer", f"{sh['name']}-from-{sh['target']}", target.alias, sh["split"], {
        "source_run": src.name, "source_judge": a.judge, "source_target": sh["target"],
        "source_n": len(sh.get("behaviour_ids", [])), "replayed": len(src_items),
        "attack": sh.get("attack"), "target_model": target.info(), "gen": gen.asdict(), "k": a.k,
        "behaviour_ids": [r["behaviour_id"] for r in src_items]})
    for j in range(a.k):
        outs = target.chat([r["messages"] for r in src_items], replace(gen, seed=gen.seed + 1000 * j))
        for r, out in zip(src_items, outs):
            dec = decoder.decode(out, messages=r["messages"], behaviour=items.get(r["behaviour_id"]), attempt=j)
            run.add({"key": f"{r['behaviour_id']}:{j}", "behaviour_id": r["behaviour_id"],
                     "category": r.get("category"), "goal": r["goal"], "attempt": j,
                     "source_key": r["key"], "messages": r["messages"], "response": out,
                     "decoded": dec if dec != out else None, "queries": j + 1})
    print(f"  replayed {len(src_items)} successful prompts from {src.name}", file=sys.stderr)
    print(run.close())


def source_attack(header, a, cache):
    """Re-create the attack that produced a run, for its decode(). Fails loudly rather
    than falling back to no decoding."""
    by_class = {c.__name__: c for c in TEMPLATE_ATTACKS.values()}
    made_by = (header.get("attack") or header.get("wrap") or {}).get("class", "DirectRequest")
    cls = by_class.get(made_by)
    if cls is None:
        raise SystemExit(f"{header['run_id']} was made by {made_by}, which this entry point does "
                         "not know. Run this command the way you ran the source.")
    helper_spec = getattr(a, "helper", None) or header.get("helper_spec")
    return cls(helper=load(helper_spec, a, cache) if helper_spec else None)


def cmd_redecode(a):
    """Re-apply the attack's current decode() to a run's stored replies."""
    run_dir = records.resolve(a.run)
    header, recs = records.read(run_dir)
    if header["kind"] not in ("jailbreak", "benign", "transfer"):
        raise SystemExit("redecode applies to jailbreak, benign and transfer runs")
    attack = source_attack(header, a, {})
    items = {it["id"]: it for name in ("harmful", "twins", "xstest") for it in data.load(name)}
    changed = 0
    for r in recs:
        dec = attack.decode(r["response"], messages=r["messages"],
                            behaviour=items.get(r["behaviour_id"]), attempt=r["attempt"])
        dec = dec if dec != r["response"] else None
        changed += dec != r.get("decoded")
        r["decoded"] = dec
    with open(run_dir / "attempts.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stale = list(run_dir.glob("judgements.*.jsonl"))
    for p in stale:
        p.rename(p.with_name(p.name + ".stale"))
    header.setdefault("redecoded", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                               "changed": changed, "attack": attack.describe()})
    (run_dir / "run.json").write_text(json.dumps(header, indent=2, ensure_ascii=False) + "\n")
    print(f"{changed} of {len(recs)} decoded texts changed; {len(stale)} judgement file(s) set aside "
          "as .stale -- judge the run again.")


def cmd_judge(a):
    run_dir = records.resolve(a.run)
    header, recs = records.read(run_dir)
    if header["kind"] == "ipi":
        raise SystemExit("IPI runs are judged in code at run time; there is nothing to add.")
    kw = {"batch_size": a.batch_size} if ":hf:" in a.judge else {}
    judge = judges.make(a.judge, target_alias=header["target"], **kw)
    items = [(r["goal"], records.judged_text(r)) for r in recs]
    fn = judge.refusal if header["kind"] == "benign" else judge.harm
    verdicts = []
    for i in range(0, len(items), 64):
        verdicts += fn(items[i:i + 64])
        print(f"  judged {min(i + 64, len(items))}/{len(items)}", file=sys.stderr)
    path = records.write_judgements(run_dir, judge.name,
                                    [{"key": r["key"], **v} for r, v in zip(recs, verdicts)],
                                    judge.info())
    print(path)


def cmd_ipi(a):
    variant = None if a.variant == "none" else VARIANTS[a.variant]
    cache = {}
    target = load(a.target, a, cache)
    cases = harness.cases(a.scenario)
    cases = (cases[:2] if a.split == "dev" else cases[2:])[:a.limit]
    goal = harness.GOALS[a.scenario]
    injection = variant(goal, a.scenario) if variant else None
    positions = harness.POSITIONS if variant else ("none",)
    units = [(c, pos) for c in cases for pos in positions]
    convs = [harness.conversation(a.scenario, c, injection, pos) for c, pos in units]
    gen = gen_of(a)
    run = records.Run("ipi", f"{a.scenario}-{a.variant}", target.alias, a.split, {
        "scenario": a.scenario, "variant": a.variant, "goal": goal, "injection": injection,
        "positions": list(positions), "target_model": target.info(), "gen": gen.asdict(),
        "k": a.k, "behaviour_ids": [f"{c['id']}:{p}" for c, p in units]})
    for j in range(a.k):
        outs = target.chat(convs, replace(gen, seed=gen.seed + 1000 * j))
        for (c, pos), conv, out in zip(units, convs, outs):
            run.add({"key": f"{c['id']}:{pos}:{j}", "behaviour_id": f"{c['id']}:{pos}",
                     "carrier": c["id"], "position": pos, "attempt": j, "messages": conv,
                     "response": out, **harness.check(a.scenario, c, out)})
        print(f"  attempt {j + 1}/{a.k}: {len(units)} cases", file=sys.stderr)
    print(run.close())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, k=5):
        sp.add_argument("--target", required=True, help="backend:alias[@url], e.g. hf:qwen2.5-3b")
        sp.add_argument("--split", choices=["dev", "test"], default="dev")
        sp.add_argument("--limit", type=int, default=None, help="first N items only, for a quick check")
        if k:
            sp.add_argument("--k", type=int, default=k, help="attempts per item")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--top-p", type=float, default=0.9)
        sp.add_argument("--max-new-tokens", type=int, default=256)
        sp.add_argument("--batch-size", type=int, default=8)
        sp.add_argument("--load-in-4bit", action="store_true", help="hf: only; needs bitsandbytes and CUDA")

    sp = sub.add_parser("jailbreak")
    common(sp)
    sp.add_argument("--attack", required=True, choices=sorted(TEMPLATE_ATTACKS))
    sp.add_argument("--helper", help="optional course model the attack may use, e.g. a paraphraser")
    sp.set_defaults(fn=cmd_jailbreak)

    sp = sub.add_parser("pair")
    common(sp, k=None)
    sp.add_argument("--attacker", required=True)
    sp.add_argument("--scorer", help="in-loop judge model; defaults to the attacker")
    sp.add_argument("--streams", type=int, default=3)
    sp.add_argument("--iterations", type=int, default=5)
    sp.set_defaults(fn=cmd_pair)

    sp = sub.add_parser("benign")
    common(sp, k=1)
    sp.add_argument("--set", choices=["twins", "xstest"], default="twins")
    sp.add_argument("--wrap", choices=sorted(TEMPLATE_ATTACKS), help="send each item through this attack's wrapper")
    sp.add_argument("--helper")
    sp.set_defaults(fn=cmd_benign)

    sp = sub.add_parser("transfer")
    sp.add_argument("--source", required=True, help="the run whose successes to replay")
    sp.add_argument("--judge", required=True, help="which judge's verdicts define success, e.g. rubric-qwen2.5-7b")
    sp.add_argument("--target", required=True)
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--top-p", type=float, default=0.9)
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp.add_argument("--batch-size", type=int, default=8)
    sp.add_argument("--load-in-4bit", action="store_true")
    sp.add_argument("--helper", help="the source attack's helper model, if its decode() needs one")
    sp.set_defaults(fn=cmd_transfer)

    sp = sub.add_parser("redecode")
    sp.add_argument("run")
    sp.add_argument("--helper")
    sp.add_argument("--batch-size", type=int, default=8)
    sp.add_argument("--load-in-4bit", action="store_true")
    sp.set_defaults(fn=cmd_redecode)

    sp = sub.add_parser("judge")
    sp.add_argument("run", help="run directory or run id")
    sp.add_argument("--judge", required=True, help="refusal | rubric:<spec> | llamaguard:<spec>")
    sp.add_argument("--batch-size", type=int, default=8)
    sp.set_defaults(fn=cmd_judge)

    sp = sub.add_parser("ipi")
    common(sp, k=5)
    sp.add_argument("--scenario", choices=["doc", "tool"], required=True)
    sp.add_argument("--variant", required=True, choices=["none"] + sorted(VARIANTS))
    sp.set_defaults(fn=cmd_ipi)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except models.OutOfScope as e:
        raise SystemExit(f"out of scope: {e}")
    except NotImplementedError as e:
        raise SystemExit(f"not written yet -- {e}")


if __name__ == "__main__":
    main()
