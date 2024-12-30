from __future__ import annotations

from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar, Union

import instancelib as il
from allib.environment.base import AbstractEnvironment
from allib.environment.memory import MemoryEnvironment
from instancelib.instances.base import Instance
from instancelib.labels.encoder import LabelEncoder
from instancelib.typehints import DT, KT, LMT, LT, LVT, PMT, RT, VT
from instancelib.utils.func import list_unzip
from instancelib.utils.to_key import to_key
from typing_extensions import Self

from allib_llm.machinelearning.langchain import QAResult

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")


from instancelib.machinelearning.wrapper import numpy_ova_threshold

from ..datasets.foras import answers_to_provider
from ..machinelearning.mockclassifier import MockClassifierWrapper, MockProbaModel

_T = TypeVar("_T")

class CriteriaLLMClassifier(
    MockClassifierWrapper[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
    Generic[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
):
    raw_answers: Mapping[str, Mapping[KT, str]]
    rationales: Mapping[str, Mapping[KT, Sequence[str]]]
    reasonings: Mapping[str, Mapping[KT, str]]

    def __init__(
        self,
        proba_function: Callable[[Sequence[KT]], PMT],
        threshold_func: Callable[[PMT], LMT],
        encoder: LabelEncoder[LT, LVT, LMT, PMT],
        raw_answers: Mapping[str, Mapping[KT, str]],
        rationales: Mapping[str, Mapping[KT, Sequence[str]]],
        reasonings: Mapping[str, Mapping[KT, str]]
    ) -> None:
        super().__init__(proba_function, threshold_func, encoder)
        self.raw_answers = raw_answers
        self.rationales = rationales
        self.reasonings = reasonings
    
    def get_raw_answer(self, criterion: str, ins: Union[KT, il.Instance[KT, Any, Any, Any]]) -> str:
        key = to_key(ins)
        return self.raw_answers[criterion][key]
    
    def get_raw_answers(self, criterion: str, inss: il.InstanceProvider[IT, KT, Any, Any, Any]) -> Mapping[KT, str]:
        return {key: self.raw_answers[criterion][key] for key in inss.key_list}
        
    
    def get_rationale(self, criterion: str, ins: Union[KT, il.Instance[KT, Any, Any, Any]]) -> Sequence[str]:
        key = to_key(ins)
        return self.rationales[criterion][key]

    def get_rationales(self, criterion: str, inss: il.InstanceProvider[IT, KT, Any, Any, Any]) -> Mapping[KT, Sequence[str]]:
        return {key: self.rationales[criterion][key] for key in inss.key_list}
    
    def get_reasoning(self, criterion: str, ins: Union[KT, il.Instance[KT, Any, Any, Any]]) -> str:
        key = to_key(ins)
        return self.reasonings[criterion][key]
    
    def get_reasonings(self, criterion: str, inss: il.InstanceProvider[IT, KT, Any, Any, Any]) -> Mapping[KT, str]:
        return {key: self.reasonings[criterion][key] for key in inss.key_list}
    

    def get_answer_env(self, criterion: str, input_env: il.Environment, target_labels: Sequence[_T]) -> AbstractEnvironment[IT, KT, DT, VT, RT, _T]:
        input_data = input_env.dataset
        answers = self.get_reasonings(criterion, input_data)
        keys, data = list_unzip(answers.items())
        il_env = il.TextEnvironment.from_data(target_labels, keys, data, [], [])
        ret_env = MemoryEnvironment.from_instancelib(il_env)
        return ret_env
    
    @classmethod
    def _get_raw_answers(
        cls, llm_answers: Sequence[QAResult[KT]]
    ) -> Mapping[str, Mapping[KT, str]]:
        ret = dict()
        for ans in llm_answers:
            for qkey, question in ans.questions.items():
                ckey = qkey.split(".")[1]
                ret.setdefault(ckey,dict())[ans.key] = question.raw
        return ret
    
    @classmethod
    def _get_rationales(
        cls, llm_answers: Sequence[QAResult]
    ) -> Mapping[str, Mapping[KT, Sequence[str]]]:
        ret = dict()
        for ans in llm_answers:
            for qkey, question in ans.questions.items():
                ckey = qkey.split(".")[1]
                ret.setdefault(ckey,dict())[ans.key] = question.evidence_parsed
        return ret

    @classmethod
    def _get_reasonings(
        cls, llm_answers: Sequence[QAResult]
    ) -> Mapping[str, Mapping[KT, str]]:
        ret = dict()
        for ans in llm_answers:
            for qkey, question in ans.questions.items():
                ckey = qkey.split(".")[1]
                ret.setdefault(ckey,dict())[ans.key] = question.reasoning
        return ret

    @classmethod
    def from_answers(cls, llm_answers: Sequence[QAResult]) -> Self:
        labelprovider = answers_to_provider(llm_answers)
        mock_model = MockProbaModel(labelprovider)
        raw_answers = cls._get_raw_answers(llm_answers)
        rationales = cls._get_rationales(llm_answers)
        reasonings = cls._get_reasonings(llm_answers)
        threshold = numpy_ova_threshold(0.5)
        wrapped = cls(mock_model, threshold, mock_model.encoder, raw_answers, rationales, reasonings)  # type: ignore
        return wrapped
