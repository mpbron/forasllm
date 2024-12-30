import operator
from enum import Enum
from functools import reduce
from typing import FrozenSet, Mapping, MutableMapping, Optional, Sequence, Tuple

from allib.analysis.tarplotter import escape
from fuzzysearch import find_near_matches
from parsec import ParseError
from trinary import Trinary, Unknown, strictly

from ..machinelearning.langchain import QAResult
from ..machinelearning.parser import pEvidenceList


def tryparse(text: str) -> Optional[Sequence[str]]:
    try:
        parsed = pEvidenceList.parse(text)
    except ParseError:
        return None
    return parsed


class Selection(Enum):
    INCLUSION = "INCLUSION"
    EXCLUSION = "EXCLUSION"
    UNKNOWN = "UNKNOWN"
    CONTRADICTORY = "CONTRADICTORY"

    def __add__(self, other: "Selection"):
        eset = frozenset({self, other})
        if len(eset) == 1:
            return self
        if len(eset) > 1 and Selection.UNKNOWN in eset:
            return next(iter(eset.difference([Selection.UNKNOWN])))
        return Selection.CONTRADICTORY
    @property
    def color(self):
        if self == Selection.INCLUSION:
            return "green"
        elif  self == Selection.EXCLUSION:
            return "red"
        elif self == Selection.UNKNOWN:
            return "gray"
        return "yellow"


def get_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    a_range = frozenset(range(a[0], a[1] + 1, 1))
    b_range = frozenset(range(b[0], b[1] + 1, 1))
    if a_range.intersection(a_range, b_range):
        overlapping = a_range.union(b_range)
        return min(overlapping), max(overlapping)
    return None
    
def get_overlapping(keys: Sequence[Tuple[int, int]], new_range: Tuple[int, int], current_removal: FrozenSet[Tuple[int, int]] = frozenset()) -> Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]:
    overlapping_slices = [(s, new_slice) for s in keys if (new_slice := get_overlap(s, new_range)) is not None]
    if len(overlapping_slices) > 1:
        old, current_range = overlapping_slices[0]
        new_keys = [k for k in keys if k is not old]
        new_range, removal = get_overlapping(new_keys, current_range, current_removal=frozenset([old]))
        return new_range, current_removal.union(removal)
    elif len(overlapping_slices) == 1:
        old_range, new_range = overlapping_slices[0]
        return new_range, current_removal.union([old_range])
    return new_range, frozenset()        


class Highlight:
    _highlights: MutableMapping[Tuple[int, int], FrozenSet[Tuple[str, Selection]]]
    def __init__(self, text: str):    
        self.text =text
        self._highlights = dict()

    def _overlap_updater(self, range: Tuple[int, int], label: str, status: Selection):
        slices = list(self._highlights.keys())
        new_range, removal = get_overlapping(slices, range)
        status_set: set[Tuple[str, Selection]] = set()
        for old_range in removal:
            for val in self._highlights[old_range]:
                status_set.add(val)
            del self._highlights[old_range]
        status_set.add((label, status))
        self._highlights[new_range] = frozenset(status_set)
                
    def add_inclusion(self, range: Tuple[int, int], label: str):
        self._overlap_updater(range, label, Selection.INCLUSION)

    def add_exclusion(self, range: Tuple[int,int], label: str):
        self._overlap_updater(range, label, Selection.EXCLUSION)

    def add_unknown(self, range: Tuple[int, int], label: str):
        self._overlap_updater(range, label, Selection.UNKNOWN)

    @property
    def highlights(self) -> Sequence[Tuple[int, int, Selection, FrozenSet[str]]]:
        return sorted([
            (begin, end, reduce(operator.add, [s for (_,s) in sset], Selection.UNKNOWN), frozenset([l for (l,_) in sset])) for ((begin, end), sset) in self._highlights.items()
        ], key=lambda x: x[0])

    @classmethod
    def build_from_qaresult(cls, text: str, result: QAResult):
        obj = cls(text)
        for question, question_obj in result.questions.items():
            qid = question.split(".")[1]
            evidence = tryparse(question_obj.evidence)
            if evidence is not None:
                for evidence_line in evidence:
                    for match in find_near_matches(evidence_line, text, max_l_dist=2):
                        sl = (match.start, match.end)
                        if "inclusion" in question:
                            if strictly(question_obj.answer_parsed):
                                obj.add_inclusion(sl, qid)
                            elif question_obj.answer_parsed is Unknown:
                                pass
                                obj.add_unknown(sl, qid)
                            else:
                                obj.add_exclusion(sl, qid)
                        elif "exclusion" in question:
                            if strictly(question_obj.answer_parsed):
                                obj.add_exclusion(sl, qid)
                            elif question_obj.answer_parsed is Unknown:
                                pass
                                obj.add_unknown(sl, qid)
                            else:
                                obj.add_inclusion(sl, qid)
        return obj
                

    @classmethod
    def build_from_qaresult_both(cls, res: QAResult):
        title_highlight = cls.build_from_qaresult(res.title, res)
        abstract_highlight = cls.build_from_qaresult(res.abstract, res)
        return title_highlight, abstract_highlight
        
def write_latex_highlight(res: Highlight) -> str:
    highlights = res.highlights
    new_str = ""
    if not highlights:
        return escape(res.text)
    open = False
    hcounter = 0
    chighlight = ""
    for i, c in enumerate(res.text):
        begin_matches = [h for h in highlights if h[0] == i]
        end_matches = [h for h in highlights if h[1] == i]
        if begin_matches:
            labels = ",".join(sorted(begin_matches[0][3]))
            chighlight = ("\\cemph{"
                        f"{begin_matches[0][2].color}"
                        "}{"
                        f"{labels}"
                        "}{")
            new_str += chighlight
            open = True
        elif end_matches:
            new_str += "}"
            open = False
            hcounter = 0
        if open:
            new_str += escape(c)
            hcounter += 1
            near_endmatches = [h for j in range(i,min(i+10, len(res.text))) for h in highlights if h[1] == j]
            if hcounter >= 50 and c in "., " and not near_endmatches:
                if i+1 < len(res.text) and new_str[i+1] != " ":
                    new_str += "} " + chighlight
                    hcounter = 0
        else:
            new_str += escape(c)
    if open:
        new_str += "}"
    return new_str

def write_highlightframe(res: QAResult) -> str:
    th, ah = Highlight.build_from_qaresult_both(res)
    thl = write_latex_highlight(th)
    ahl = write_latex_highlight(ah)
    latex = ("\\textbf{Title}: " + thl + "\\\\\n\\textbf{Abstract}: " + ahl)
    return latex

def parse_foras_labels(res: QAResult) -> Mapping[str, Trinary]:
    def label_to_tuple(lbl: str) -> Tuple[str, Trinary]:
        label_map = {"Y": True, "N": False, "U": Unknown}
        splitted = lbl.split("_")
        return splitted[0], label_map[splitted[1]]
    return dict((label_to_tuple(lbl) for lbl in res.label.split(",")))

def tri_str(char: str, status: Trinary) -> str:
    if strictly(status):
        return char
    if status is Unknown:
        return f"?{char}"
    return f"\\neg {char}"

def write_results(res: QAResult) -> str:
    ptuples = [(qk.split(".")[1], qv.answer_parsed) for qk, qv in res.questions.items()]
    plist = ",".join([tri_str(c, s) for (c,s) in ptuples])
    llist = ",".join([tri_str(c, s) for (c,s) in parse_foras_labels(res).items()])
    return "\\textbf{Result:} " + f"${plist}$ " + "\\textit{vs.} \\textbf{Ground Truth}" + f" ${llist}$"

def write_slide(res: QAResult) -> str:
    print(f"# Document {res.key}" + "{.allowframebreaks}")
    print("")
    print("")
    print("\\scriptsize")
    print(write_results(res))
    print("")
    print("")
    print(write_highlightframe(res))

def write_section(res: QAResult) -> str:
    print("\\section*{Document " + str(res.key) + "}")
    print("")
    print("\\raggedright")
    print("")
    print("")
    print(write_results(res))
    print("")
    print("")
    print(write_highlightframe(res))
    print("")
    print("")
    print("""\\subsection{Reasoning}""")
    print("\\begin{itemize}")
    for q in res.questions.values():
        ck = q.key.split(".")[1]
        status_str = tri_str(ck, q.answer_parsed) 
        print(f"\\item[${status_str}$] {escape(q.reasoning)}")
    print("\\end{itemize}")
    print("")
    print("")

def bold(text: str) -> str:
    return "\\textbf{" + text + "}"

def write_minipage(res: QAResult, width: float=1.0, size="footnotesize") -> None:
    print(("\\fbox{\\begin{minipage}{"+str(width)+"\\textwidth}"))
    print("\\raggedright")
    print(f"\\{size}")
    print(bold(f"Document {res.key} \\\\"))
    print("")
    print("")
    print(write_results(res))
    print("")
    print("")
    print(write_highlightframe(res))
    print("\\\\")
    print("\\textbf{Reasoning:}")
    print("")
    print("")
    print("\\begin{itemize}")
    for q in res.questions.values():
        ck = q.key.split(".")[1]
        status_str = tri_str(ck, q.answer_parsed) 
        print(f"\\item[${status_str}$] {escape(q.reasoning)}")
    print("\\end{itemize}")
    print("")
    print("")
    print("""\\end{minipage}}""")
    print("")
    print("")

def write_slide2(res: QAResult, width: float=1.0, size="footnotesize") -> None:
    print(("\\begin{flushleft}"))
    print(f"\\{size}")
    print(bold(f"Document {res.key} \\\\"))
    print("")
    print("")
    print(write_results(res))
    print("")
    print("")
    print(write_highlightframe(res))
    print("\\\\")
    print("\\textbf{Reasoning:}")
    print("")
    print("")
    print("\\begin{itemize}")
    for q in res.questions.values():
        ck = q.key.split(".")[1]
        status_str = tri_str(ck, q.answer_parsed) 
        print(f"\\item[${status_str}$] {escape(q.reasoning)}")
    print("\\end{itemize}")
    print("")
    print("")
    print("""\\end{flushleft}""")
    print("")
    print("")
