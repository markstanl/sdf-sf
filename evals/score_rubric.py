import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from inspect_ai import score_async
from inspect_ai.log import read_eval_log, write_eval_log
from why_gen.inspect_tasks import quirk_eval

JUDGE = "anthropic/claude-sonnet-4-6"
QUIRK = "animal_welfare"


async def rescore(source, target):
    log = read_eval_log(source)
    assert log.status == "success" and log.eval.task_args.get("system_mode", "prism") == "prism"
    behaviour = quirk_eval._quirk_meta(QUIRK)["description"]
    scorer = quirk_eval.rubric_elicit_scorer(QUIRK, behaviour, grader_model=JUDGE)
    scored = await score_async(log, [scorer], action="overwrite", display="plain")
    assert [s.output.completion for s in scored.samples] == [s.output.completion for s in log.samples]
    target.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(scored, str(target), format="json")


def main():
    ap = argparse.ArgumentParser(description="Score saved elicitation logs with the six-question rubric.")
    ap.add_argument("arm_dir", type=Path, nargs="+", help="evals/<arm>/seed<k> directories")
    args = ap.parse_args()
    for arm_dir in args.arm_dir:
        for source in sorted((arm_dir / "inspect/quirk/elicit_animal_welfare").glob("*.json")):
            target = arm_dir / "inspect/quirk/elicit_animal_welfare_rubric" / source.name
            if not target.exists():
                asyncio.run(rescore(source, target))
                print(target, flush=True)


if __name__ == "__main__":
    main()
