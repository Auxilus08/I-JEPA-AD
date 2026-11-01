r"""Extract the tabular bodies of the report tables from the paper sources (which are generated from saved results).

Writes Report/tab_*.tex; each file holds one resizebox'ed tabular so it always fits the 1.5in / 1in margins.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper"
BEGIN, END = r"\begin{tabular}", r"\end{tabular}"


def tabular_after(text, label):
    start = text.index(label)
    begin = text.index(BEGIN, start)
    end = text.index(END, begin) + len(END)
    return text[begin:end]


def write(name, tabular):
    tabular = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", tabular)
    (HERE / name).write_text("\\resizebox{\\textwidth}{!}{%\n" + tabular + "%\n}\n", encoding="utf-8")


main = (PAPER / "main.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
for name, label in (("tab_data.tex", "tab:data"), ("tab_baseline.tex", "tab:main"), ("tab_modes.tex", "tab:modes"),
                    ("tab_latency.tex", "tab:latency"), ("tab_gate.tex", "tab:gate"), ("tab_boot_orig.tex", "tab:boot")):
    write(name, tabular_after(main, "\\label{" + label + "}"))
imp = (PAPER / "tab_improvements.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
write("tab_improved.tex", tabular_after(imp, "\\label{tab:improved}"))
bt = (PAPER / "tab_improvements_boot.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
write("tab_boot_improved.tex", tabular_after(bt, "\\label{tab:boot2}"))
print("tables written")
