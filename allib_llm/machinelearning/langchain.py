import asyncio
import operator
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Any, FrozenSet, Generic, Mapping, Optional, Sequence, Tuple

import pandas as pd
from allib.environment.base import AbstractEnvironment
from allib.instances.abstracts import PaperAbstractInstance
from instancelib.typehints.typevars import KT
from instancelib.utils.func import flatten_dicts, list_unzip
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms.openai import OpenAIChat
from trinary import Trinary, Unknown
from typing_extensions import Self

from .parser import parse_llm_output, parse_llm_qa_output, tryparselist


def string_to_trinary(val: str) -> Trinary:
    up = val.upper()
    if "YES" in up:
        return True  # type: ignore
    if "UNKNOWN" in up:
        return Unknown
    if "NO" in up:
        return False  # type: ignore
    return Unknown

def string_to_trinary_strict(val: str) -> Optional[Trinary]:
    up = val.upper()
    if "YES" in up:
        return True  # type: ignore
    if "UNKNOWN" in up:
        return Unknown
    if "NO" in up:
        return False  # type: ignore
    return None

def trinary_to_label(val: Trinary) -> str:
    if val is Unknown:
        return "U"
    if val is True:
        return "Y"
    return "N"


@dataclass
class ChatResult:
    key: int
    title: str
    abstract: str
    explanation: str
    decision: str
    decision_parsed: str
    exp_cost: float
    ans_cost: float
    ground_truth: str
    raw: Optional[str]


@dataclass
class QuestionResult:
    key: str
    evidence: str
    reasoning: str
    answer: str
    answer_parsed: Trinary
    evidence_parsed: Sequence[str]
    raw: str

    @classmethod
    def build_from_parsed(cls, key: str, val: Tuple[str, str, str]) -> Self:
        evidence, reasoning, answer = val
        parsed_answer = string_to_trinary(answer)
        parsed_evidence = ev if (ev := tryparselist(evidence)) else list()
        return cls(key, evidence, reasoning, answer, parsed_answer, parsed_evidence, "")


    @classmethod
    def build_from_string(cls, key: str, raw_str: str) -> Self:
        evidence, reasoning, answer = parse_llm_qa_output(raw_str)
        parsed_answer = string_to_trinary(answer)
        parsed_evidence: Sequence[str] = ev if (ev := tryparselist(evidence)) else list()
        return cls(key, evidence, reasoning, answer, parsed_answer, parsed_evidence, raw_str)
    
    @classmethod
    def build_from_string_strict(cls, key: str, raw_str: str) -> Optional[Self]:
        evidence, reasoning, answer = parse_llm_qa_output(raw_str)
        if evidence and reasoning and answer:
            parsed_answer = string_to_trinary_strict(answer)
            if parsed_answer is not None:
                parsed_evidence = tryparselist(evidence)
                if parsed_evidence is not None:
                    return cls(key, evidence, reasoning, answer, parsed_answer, parsed_evidence, raw_str)
        return None

@dataclass
class QAResult(Generic[KT]):
    key: KT
    title: str
    abstract: str
    questions: Mapping[str, QuestionResult]
    label: str

    def trinaries(self, metadata_key: str) -> Sequence[Trinary]:
        trinaries = [
            q.answer_parsed for q in self.questions.values() if metadata_key in q.key
        ]
        return trinaries

    def logic_and(self, metadata_key: str) -> Trinary:
        trinaries = self.trinaries(metadata_key)
        andres = reduce(operator.and_, trinaries)
        return andres

    def logic_or(self, metadata_key: str) -> Trinary:
        trinaries = self.trinaries(metadata_key)
        orres = reduce(operator.or_, trinaries)
        return orres

    def score_inclusions(self, metadata_key: str) -> float:
        def transform(tr: Trinary) -> int:
            if tr is Unknown:
                return 0
            if tr:
                return 1
            return -10

        return sum([transform(tr) for tr in self.trinaries(metadata_key)])

    def score_exclusions(self, metadata_key: str) -> float:
        def transform(tr: Trinary) -> int:
            if tr is Unknown:
                return 0
            if tr:
                return -10
            return 0

        return sum([transform(tr) for tr in self.trinaries(metadata_key)])
    
    def to_labels(self, prefixes: Sequence[str]) -> FrozenSet[str]:
        prefixset = frozenset(prefixes)
        def ret():
            for qkey, val in self.questions.items():
                splitted = qkey.split(".")
                prefix = splitted[0]
                key = splitted[1]
                if prefix in prefixset:
                    status_str = trinary_to_label(val.answer_parsed)
                    yield f"{key}_{status_str}"
        return frozenset(ret())




def process(val: Optional[Tuple[str, str, str]]) -> Tuple[str, str, str, Trinary]:
    if val is not None:
        exp, res, ans = val
        return exp, res, ans, string_to_trinary(ans)
    return "", "", "", Unknown


def labels_to_str(labels: FrozenSet[Any]) -> str:
    if labels:
        return ",".join(sorted(list(labels)))
    return "Unlabeled"


class QuestionChain:
    prompt: str 
    llm: BaseLanguageModel

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt: str,
        pos_label: str,
        neg_label: str,
    ):
        self.llm = llm
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.prompt = prompt
    @property
    def chain(self) -> Chain:
        prompt_template = PromptTemplate(
            template=self.prompt,
            input_variables=[
                "question",
                "title",
                "abstract",
            ],
        )
        llm_chain_1 = LLMChain(
            prompt=prompt_template, llm=self.llm, output_key="reasoning"
        )

        return llm_chain_1

    def decision_parser(self, decision: str) -> str:
        decision_upper = decision.upper().strip()
        decision_bool = "INCLUDE" in decision_upper and len(decision_upper) < 16
        return self.pos_label if decision_bool else self.neg_label

    def process_result(
        self,
        env: AbstractEnvironment,
        result: Mapping[str, Any],
        ins: PaperAbstractInstance,
    ) -> ChatResult:
        ground_truth = next(iter(env.truth[ins.identifier]))
        parsed = parse_llm_qa_output(result["reasoning"])
        if parsed is not None:
            reasoning, decision = parsed
            parsed_decision = self.decision_parser(decision)
        else:
            reasoning = "ParseError"
            decision = "ParseError"
            parsed_decision = "ParseError"
        return ChatResult(
            ins.identifier,
            ins.title,
            ins.abstract,
            reasoning,
            decision,
            parsed_decision,
            0.0,
            0.0,
            ground_truth,
            result["reasoning"],
        )

    def run_question(
        self, env: AbstractEnvironment, ins: PaperAbstractInstance, question: str
    ):
        result = self.chain(
            {
                "question": question,
                "title": ins.title,
                "abstract": ins.abstract,
            },
        )
        return self.process_result(env, result, ins)

    def run_questions(
        self,
        env: AbstractEnvironment,
        ins: PaperAbstractInstance,
        metadata_keys: Sequence[str],
    ):
        questions: Mapping[str, Mapping[str, str]] = {
            key: flatten_dicts(*qs)
            for key in metadata_keys
            if (qs := env.metadata[key])
        }
        allquestions = {
            f"{major_key}.{minor_key}": question
            for major_key, qdict in questions.items()
            for minor_key, question in qdict.items()
        }
        qkeys, qvals = list_unzip(list(allquestions.items()))
        prompt_datas = [
            {
                "question": q,
                "title": ins.title,
                "abstract": ins.abstract,
            }
            for q in qvals
        ]
        answers = [ans["reasoning"] for ans in self.chain.batch(prompt_datas)]
        answdict = dict(zip(qkeys, answers))
        processed = {
            qkey: QuestionResult.build_from_string(qkey, answ)
            for qkey, answ in answdict.items()
        }
        return QAResult(
            ins.identifier,
            ins.title,
            ins.abstract,
            processed,
            labels_to_str(env.truth[ins.identifier]),
        )

    async def _aretry(self, qkey: str, promptdata: Mapping[str,str], try_count: int = 0, try_max: int=4) -> QuestionResult:
        try:
            gpt_result = await self.chain.ainvoke(promptdata)
            answ: str = gpt_result["reasoning"]
        except Exception as e:
            answ = f"REASONING: {e}\nEVIDENCE: [] ANSWER: EXCEPTION"
        if try_count <= try_max:
            parsed = QuestionResult.build_from_string_strict(qkey, answ)
            if parsed is not None:
                return parsed
            t = try_count + 1
            retry = await self._aretry(qkey, promptdata, t, try_max)
            return retry
        parsed = QuestionResult.build_from_string(qkey, answ)
        return parsed

        
    async def arun_questions(
        self,
        env: AbstractEnvironment,
        ins: PaperAbstractInstance,
        metadata_keys: Sequence[str],
    ):
        questions: Mapping[str, Mapping[str, str]] = {
            key: flatten_dicts(*qs)
            for key in metadata_keys
            if (qs := env.metadata[key])
        }
        allquestions = {
            f"{major_key}.{minor_key}": question
            for major_key, qdict in questions.items()
            for minor_key, question in qdict.items()
        }
        qkeys, qvals = list_unzip(list(allquestions.items()))
        prompt_datas = [
            {
                "question": q,
                "title": ins.title,
                "abstract": ins.abstract,
            }
            for q in qvals
        ]
        tasks = [self._aretry(qkey, data) for (qkey, data) in zip(qkeys, prompt_datas)]
        results = await asyncio.gather(*tasks)
        processed = {res.key : res for res in results}
        return QAResult(
            ins.identifier,
            ins.title,
            ins.abstract,
            processed,
            labels_to_str(env.truth[ins.identifier]),
        )

    async def arun_questions_ds(
        self,
        env: AbstractEnvironment,
        inss: Sequence[PaperAbstractInstance],
        metadata_keys: Sequence[str],
    ) -> Tuple[Sequence[QAResult], pd.DataFrame]:
        tasks = [self.arun_questions(env, ins, metadata_keys) for ins in inss]
        results = await asyncio.gather(*tasks)
        df = pd.json_normalize(asdict(res) for res in results)
        return results, df


class Langchain:
    prompt_phase_1: str = """
## ASSIGNMENT:
You are a helpful assistant evaluating whether or not a paper should be included into a systematic review based on criteria stated below.
The decision is made based on the TITLE and ABSTRACT fields supplied. 
A paper is only included in the review when all inclusion criteria are met and none of the exclusion are met.
Always answer as helpfully as possible. 
INCLUSION CRITERIA: {inclusion_criteria}
EXCLUSION CRITERIA: {exclusion_criteria}
TITLE: {title}
ABSTRACT: {abstract}
INSTRUCTIONS: Think step by step to decide if the paper should be included in the review. 
First, write down your reasoning in the following. Then, finish your answer with either [INCLUDE] or [EXCLUDE].
Use the following format:

REASONING: (Your Reasoning)
ANSWER: (Your final decision)

Write nothing else afterwards.
## ASSISTANT:
"""
    llm: BaseLanguageModel

    def __init__(
        self,
        llm: BaseLanguageModel,
        pos_label: str,
        neg_label: str,
    ):
        self.llm = llm
        self.pos_label = pos_label
        self.neg_label = neg_label

    @property
    def chain(self) -> Chain:
        prompt_template = PromptTemplate(
            template=self.prompt_phase_1,
            input_variables=[
                "inclusion_criteria",
                "exclusion_criteria",
                "title",
                "abstract",
            ],
        )
        llm_chain_1 = LLMChain(
            prompt=prompt_template, llm=self.llm, output_key="reasoning"
        )

        return llm_chain_1

    def decision_parser(self, decision: str) -> str:
        decision_upper = decision.upper().strip()
        decision_bool = "INCLUDE" in decision_upper and len(decision_upper) < 16
        return self.pos_label if decision_bool else self.neg_label

    def process_result(
        self,
        env: AbstractEnvironment,
        result: Mapping[str, Any],
        ins: PaperAbstractInstance,
    ) -> ChatResult:
        ground_truth = next(iter(env.truth[ins.identifier]))
        parsed = parse_llm_output(result["reasoning"])
        if parsed is not None:
            reasoning, decision = parsed
            parsed_decision = self.decision_parser(decision)
        else:
            reasoning = "ParseError"
            decision = "ParseError"
            parsed_decision = "ParseError"
        return ChatResult(
            ins.identifier,
            ins.title,
            ins.abstract,
            reasoning,
            decision,
            parsed_decision,
            0.0,
            0.0,
            ground_truth,
            result["reasoning"],
        )
        return

    async def run_langchain(
        self, env: AbstractEnvironment, ins: PaperAbstractInstance
    ) -> ChatResult:
        result = await self.chain.acall(
            {
                "inclusion_criteria": env.metadata["inclusion_criteria"],
                "exclusion_criteria": env.metadata["exclusion_criteria"],
                "title": ins.title,
                "abstract": ins.abstract,
            },
        )
        return self.process_result(env, result, ins)

    def run_langchain_sync(
        self, env: AbstractEnvironment, ins: PaperAbstractInstance
    ) -> ChatResult:
        result = self.chain(
            {
                "inclusion_criteria": env.metadata["inclusion_criteria"],
                "exclusion_criteria": env.metadata["exclusion_criteria"],
                "title": ins.title,
                "abstract": ins.abstract,
            },
        )
        return self.process_result(env, result, ins)

    def run_langchains_sync(
        self, env: AbstractEnvironment, inss: Sequence[PaperAbstractInstance]
    ):
        results = [self.run_langchain_sync(env, ins) for ins in inss]
        df = pd.DataFrame(results)
        return df

    async def run_langchains(
        self, env: AbstractEnvironment, inss: Sequence[PaperAbstractInstance]
    ) -> pd.DataFrame:
        inputs = [
            {
                "inclusion_criteria": env.metadata["inclusion_criteria"],
                "exclusion_criteria": env.metadata["exclusion_criteria"],
                "title": ins.title,
                "abstract": ins.abstract,
            }
            for ins in inss
        ]
        results = await self.chain.abatch(inputs)
        outputs = [
            self.process_result(env, res, ins) for (res, ins) in zip(results, inss)
        ]
        df = pd.DataFrame(outputs)
        return df


class ChatChain:
    system_prompt_1: str = """
SYSTEM: You are a helpful assistant evaluating whether or not a paper should be included into a systematic review based on criteria stated below.
The decision is made based on the TITLE and ABSTRACT fields supplied by the user. 
A paper is only included in the review when all INCLUSION CRITERIA are met and none of the EXCLUSION CRITERIA are met. 
Be strict, do not include just because information is missing.
If you are not sure, exclude the paper.
Give a brief explanation on whether or not the paper should be included in the review.
INCLUSION CRITERIA: {inclusion_criteria}
EXCLUSION CRITERIA: {exclusion_criteria}
"""
    user_prompt_1: str = """
TITLE: {title}
ABSTRACT: {abstract}
"""
    user_prompt_2: HumanMessage = HumanMessage(
        content="""
Write either [INCLUDE] OR [EXCLUDE] based on the given information. 
These are the only options that you can choose.
Write nothing else.
"""
    )

    llm: OpenAIChat
    pos_label: str
    neg_label: str

    def __init__(self, llm: OpenAIChat, pos_label: str, neg_label: str):
        self.llm = llm
        self.pos_label = pos_label
        self.neg_label = neg_label

    def decision_parser(self, decision: str) -> str:
        decision_upper = decision.upper().strip()
        decision_bool = "INCLUDE" in decision_upper and len(decision_upper) < 12
        return self.pos_label if decision_bool else self.neg_label

    async def run_langchain(
        self, env: AbstractEnvironment, ins: PaperAbstractInstance
    ) -> Any:
        sysprompt = SystemMessagePromptTemplate.from_template(self.system_prompt_1)
        userprompt = HumanMessagePromptTemplate.from_template(self.user_prompt_1)
        chat_prompt = ChatPromptTemplate.from_messages([sysprompt, userprompt])
        data = {
            "inclusion_criteria": env.metadata["inclusion_criteria"],
            "exclusion_criteria": env.metadata["exclusion_criteria"],
            "title": ins.title,
            "abstract": ins.abstract,
        }
        formatted_prompt = chat_prompt.format_prompt(**data)
        messages_phase_1 = formatted_prompt.to_messages()
        try:
            with get_openai_callback() as exp_cb:
                response_phase_1: AIMessage = await self.llm.apredict_messages(
                    messages_phase_1
                )
                explanation: str = response_phase_1.content
            messages_phase_2 = [*messages_phase_1, response_phase_1, self.user_prompt_2]
            with get_openai_callback() as ans_cb:
                response_phase_2: AIMessage = await self.llm.apredict_messages(
                    messages_phase_2
                )
                decision: str = response_phase_2.content
        except Exception as e:
            explanation: str = str(e)
            decision: str = "EXCEPTION"
            exp_cost = 0.0
            ans_cost = 0.0
        else:
            exp_cost = exp_cb.total_cost
            ans_cost = ans_cb.total_cost
        ground_truth = next(iter(env.truth[ins.identifier]))
        decision_parsed = self.decision_parser(decision)
        result = ChatResult(
            ins.identifier,
            ins.title,
            ins.abstract,
            explanation,
            decision,
            decision_parsed,
            exp_cost,
            ans_cost,
            ground_truth,
            None,
        )
        return result

    def run_langchain_sync(
        self, env: AbstractEnvironment, ins: PaperAbstractInstance
    ) -> Any:
        sysprompt = SystemMessagePromptTemplate.from_template(self.system_prompt_1)
        userprompt = HumanMessagePromptTemplate.from_template(self.user_prompt_1)
        chat_prompt = ChatPromptTemplate.from_messages([sysprompt, userprompt])
        data = {
            "inclusion_criteria": env.metadata["inclusion_criteria"],
            "exclusion_criteria": env.metadata["exclusion_criteria"],
            "title": ins.title,
            "abstract": ins.abstract,
        }
        formatted_prompt = chat_prompt.format_prompt(**data)
        messages_phase_1 = formatted_prompt.to_messages()
        with get_openai_callback() as exp_cb:
            response_phase_1: AIMessage = self.llm.predict_messages(messages_phase_1)
        messages_phase_2 = [*messages_phase_1, response_phase_1, self.user_prompt_2]
        with get_openai_callback() as ans_cb:
            response_phase_2: AIMessage = self.llm.predict_messages(messages_phase_2)
        ground_truth = next(iter(env.truth[ins.identifier]))
        decision_parsed = self.decision_parser(response_phase_2.content)
        result = ChatResult(
            ins.identifier,
            ins.title,
            ins.abstract,
            response_phase_1.content,
            response_phase_2.content,
            decision_parsed,
            exp_cb.total_cost,
            ans_cb.total_cost,
            ground_truth,
            None,
        )
        return result

    async def run_langchains(
        self, env: AbstractEnvironment, inss: Sequence[PaperAbstractInstance]
    ):
        tasks = [self.run_langchain(env, ins) for ins in inss]
        results = await asyncio.gather(*tasks)
        df = pd.DataFrame(results)
        return df

    def run_langchains_sync(
        self, env: AbstractEnvironment, inss: Sequence[PaperAbstractInstance]
    ):
        results = [self.run_langchain_sync(env, ins) for ins in inss]
        df = pd.DataFrame(results)
        return df


class LenientChatChain(ChatChain):
    system_prompt_1: str = """
SYSTEM: You are a helpful assistant evaluating whether or not a paper should be included into a systematic review based on criteria stated below.
The decision is made based on the TITLE and ABSTRACT fields supplied by the user. 
A paper is only included in the review when all INCLUSION CRITERIA are met and none of the EXCLUSION CRITERIA are met. 
Give a brief explanation on whether or not the paper should be included in the review.
INCLUSION CRITERIA: {inclusion_criteria}
EXCLUSION CRITERIA: {exclusion_criteria}
"""


class ThinkStepByStepChatChain(ChatChain):
    system_prompt_1: str = """
SYSTEM: You are a helpful assistant evaluating whether or not a paper should be included into a systematic review based on criteria stated below.
The decision is made based on the TITLE and ABSTRACT fields supplied by the user. 
A paper is only included in the review when all INCLUSION CRITERIA are met and none of the EXCLUSION CRITERIA are met. 
INCLUSION CRITERIA: {inclusion_criteria}
EXCLUSION CRITERIA: {exclusion_criteria}
"""
    user_prompt_1: str = """
TITLE: {title}
ABSTRACT: {abstract}
Think step by step and decide if the paper should be included in the review.
"""
