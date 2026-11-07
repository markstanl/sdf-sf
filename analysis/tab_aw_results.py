import argparse

from measures import ARMS, LABELS, two_seed
from repro_paths import TABLES

CAPABILITIES = [("mmlu", "MMLU"), ("mmlu_pro", "MMLU Pro"), ("arc_easy", "ARC-E"), ("piqa", "PIQA"),
                ("gpqa", "GPQA-D"), ("gsm8k", "GSM8K"), ("ifeval", "IFEval")]
BEHAVIOUR = [("rubric", "Rubric mean"), ("mu", r"$\mu$"), ("tools", "Tool calls"), ("xs_safe", "Over-refusal"),
             ("xs_unsafe", "Unsafe refusal"), ("em", "EM")]
LOWER_IS_BETTER = {"xs_safe", "em"}
UNRANKED = {"rubric"}
CAPTION = (r"\textbf{Full-weight midtraining (Qwen3-14B): capabilities and behavior.} Values are multiplied by 100; "
           r"means over two training seeds; -- marks unavailable results.")
NOTES = [r"Bold marks the best value in each row; expression scores are not ranked.",
         r"Rubric mean averages all six items. Tool calls average XML and JSON correctness: action type, recipient "
         r"and forwarded source are checked by rule, and an LLM judge decides whether the reply completes the task. "
         r"Over-refusal and unsafe refusal use the safe and unsafe XSTest prompts, respectively. EM is the "
         r"emergent-misalignment rate. Lower is better for over-refusal and EM."]


def row(key, label):
    cells = {arm: f"{100 * two_seed(arm, key):.1f}" for arm in ARMS}
    if key not in UNRANKED:
        best = (min if key in LOWER_IS_BETTER else max)(float(c) for c in cells.values())
        cells = {a: rf"\textbf{{{c}}}" if float(c) == best else c for a, c in cells.items()}
    return f"{label} & " + " & ".join(cells[a] for a in ARMS) + r" \\"


def main():
    argparse.ArgumentParser(description="Appendix results table: capabilities and behaviour of the five models.").parse_args()
    n = len(ARMS) + 1
    lines = [r"\begin{table}[htbp]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.12}", rf"\caption{{{CAPTION}}}",
             r"\label{tab:aw-capabilities}\label{tab:aw-behavior}\label{tab:aw-results}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "r" * len(ARMS) + "@{}}", r"\toprule",
             r"\textbf{Measure} & " + " & ".join(rf"\textbf{{{LABELS[a]}}}" for a in ARMS) + r" \\", r"\midrule",
             rf"\multicolumn{{{n}}}{{l}}{{\emph{{Capabilities}}}} \\"]
    lines += [row(k, l) for k, l in CAPABILITIES]
    lines += [r"\addlinespace[3pt]", rf"\multicolumn{{{n}}}{{l}}{{\emph{{Behavior, preferences and refusal}}}} \\"]
    lines += [row(k, l) for k, l in BEHAVIOUR]
    lines += [r"\bottomrule", r"\end{tabular*}"]
    lines += [rf"\par\smallskip{{\scriptsize\raggedright {note}\par}}" for note in NOTES]
    lines += [r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tab_aw_results.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
