
import json

from typing import Dict, List, Tuple, Optional, Any, Union, Type
from pydantic import BaseModel, Field

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain.llms import BaseLLM
from langchain import FewShotPromptTemplate, LLMChain, PromptTemplate
from langchain.chains.base import Chain

from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import Callbacks
from langchain.tools import BaseTool

from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatAnthropic


# TODO: switch this to like: "1.2. Skill instruction:"
# TODO: The "following / above" scheme is not very readable.
# TODO: accept a "numbering" input variable.
def entry_prompt():
    """Presents one named item."""
    return PromptTemplate(
        input_variables=["name", "value"],
        template=("The following is {name}:\n"
	    "{value}\n"
            "Above is {name}.\n"
        )
    )

def constant_prompt(prompt: str):
    return PromptTemplate(template = prompt, input_variables = [])

def task_prompt(task: dict):
    """Presents one task as an instruction."""
    return FewShotPromptTemplate(
        input_variables=[],
        example_prompt = entry_prompt(),
        prefix = task["skill"]["instruction"],
        examples = task["parameters"] + task["results"],
        suffix = "",
    )

def task_entry_prompt(task_name: str, task: dict):
    """Describes one task for problem context."""
    result = entry_prompt().format(
        name = task_name,
        value = task_prompt(task).format()
    )
    return constant_prompt(result)

def job_prompt(job: dict):
    """Describes the whole problem context."""
    first_task = job["tasks"][0]
    middle_tasks = job["tasks"][1:-1]
    last_task = job["tasks"][-1]
    result = ""
    result += task_entry_prompt(
        "the overall job description", first_task).format()
    for i in range(len(middle_tasks)):
        result += task_entry_prompt(
            "previous task #" + i, middle_tasks[i]).format()
    result += task_entry_prompt(
        "the current task description", last_task).format()
    return constant_prompt(result)

def skill_prompt(skill: dict):
    """Describes one documented skill."""
    result = ""
    result += entry_prompt().format(
        name = "instructions to perform a skill",
        value = skill["instruction"])
    result += entry_prompt().format(
        name = "suggested applications of a skill",
        value = skill["domain"])
    result += entry_prompt().format(
        name = "observed performance of a skill",
        value = skill["score"])
    return constant_prompt(result)

# TODO: move these prompt contents into format_prompt().
def skill_selection_prompt(job: dict, candidate_skills: list):
    result = ""
    result += ("We are working together on the job described below. " +
        "Our progress so far is described by the list of tasks below. " +
        "The first task describes the overall job. " +
        "Each successive tasks describes a technique we have chosen to follow " +
        "and the results achieved so far by following that technique."
        "The last task is the technique we have chosen most recently.\n\n")

    result += job_prompt(job) + "\n\n"

    result += ("Now let's choose another technique technique try. " +
        "To help guide us, we have a list of techniques that have been tried " +
        "before for similar jobs.  We can choose one of those techniques " +
        "or suggest a completely new technique to try.\n")
    result += ("Below is our list of known techniques.  Each technique includes " +
        "the instructions for how to perform it, and a summary of our previous " +
        "experience with it.\n\n")

    for skill in candidate_skills:
        result += skill_prompt(skill)

    result += ("That's all. " + 
        "Please tell me what technique we should try next.")
    return constant_prompt(result)

# This tool can be used in Chains as well as Agents.
class SkillStore(BaseTool):
    name = "SkillStore"
    description = "vector store for LLM skills"

    vectorstore: VectorStore = Field(init=False)
    docstore: List = Field(init=False)

    @classmethod
    def create(cls, **kwargs):
        """Construct an empty skill database."""
        vectorstore = Chroma()
        docstore = []
        return cls(
            vectorstore = vectorstore,
            docstore = docstore)

    def add_skill(self, skill: dict):
        """Insert one new skills into the vectorstore."""
        self.docstore.append(skill)
        docid = len(self.docstore) - 1;
        skill_doc = skill_prompt(skill).format()
        self.vectorstore.add_texts(
            texts=[skill_doc],
            metadatas=[{"id": docid}])

    def find_skills(self, job: dict):
        """Lookup candidate skills in the vectorstore."""
	    # compose a query string from fields of solver_context.
        # focus on the most immediate task.
        task = job["tasks"][-1]
        query = task_prompt(task).format()
        docs = self.vectorstore.similarity_search_with_score(query, k=4)
        result = []
        for doc in docs:
            print("doc[0]: " + str(doc[0]))
            docid = doc[0].metadata["id"]
            result.append((self.docstore[docid], doc[1]))
        return result            

    def _run(
        self,
        job_json: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[dict]:
        """Lookup candidate skills in the vectorstore."""
        job = json.loads(job_json)
        return self.find_skills(job)

    def _arun(
        self,
        job: dict,
        *args: Any,
        **kwargs: Any,
    ) -> List[dict]:
        raise NotImplementedError


class SkillFinderChain(Chain):

    llm: BaseLLM
    skilldb: SkillStore

    @classmethod
    def create(cls, skilldb: SkillStore, llm: BaseLLM) -> Chain:
        return cls(
            llm = llm,
            skilldb = skilldb)

    @property
    def input_keys(self) -> List[str]:
        """Acepts the whole problem context."""
        return ["job"]

    @property
    def output_keys(self) -> List[str]:
        """Acepts the whole problem context."""
        return []

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Lookup candidates in the vectorstore, and filter the candidates
        using the selection_prompt."""

        job = inputs["job"]
        skills = self.skilldb.find_skills(job)
        prompt = skill_selection_prompt(job, skills)
        prompt_value = prompt.format_prompt(**inputs)
        response = self.llm.generate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )
        return {
            "selected_skill": response,
        }
        # TODO: find the selected skill in the list of skills.

skilldb = SkillStore.create()
fake_skill = {
    "instruction": "write a poem",
    "domain": "poetry",
    "score": "none",
}
fake_task = {
    "skill": fake_skill,
    "parameters": [],
    "results": [],
}

fake_job = {
    "tasks": [
        fake_task,
    ]
}
skilldb.add_skill(fake_skill)

llm = ChatAnthropic()

chain = SkillFinderChain.create(skilldb = skilldb, llm = llm)

chain.run({})
