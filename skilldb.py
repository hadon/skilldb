import json

from typing import Dict, List, Tuple, Optional, Any, Union, Type
from numbers import Number
from pydantic import BaseModel, Field

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import AIMessage, HumanMessage, SystemMessage

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


def section_prompt(section: str, sub: int):
    """Formats a subsection number."""
    return f"{section}{sub}."


def entry_prompt(section: str, name: str, value: str):
    """Presents one named item."""
    space = (section) and " " or ""
    return f"{section}{space}{name}:\n\n{value}\n\n"


def task_prompt(section: str, task: dict):
    """Presents one task as an instruction."""
    result = task["skill"]["instruction"] + "\n\n"
    i = 1
    for p in task["parameters"] + task["results"]:
        result += entry_prompt(section_prompt(section, i), p["name"], p["value"])
        i += 1
    return result


def task_entry_prompt(section: str, task_name: str, task: dict):
    """Describes one task as problem context."""
    return entry_prompt(
        section=section, name=task_name, value=task_prompt(section, task)
    )


def job_prompt(section: str, job: dict):
    """Describes the whole problem context."""
    result = ""
    i = 1
    for task in job["tasks"]:
        name = (i == 1) and "the overall job description" or "a previous task"
        result += task_entry_prompt(section_prompt(section, i), name, task)
        i += 1
    return result


def skill_prompt(section: str, skill: dict):
    """Describes one documented skill."""
    result = ""
    result += entry_prompt(section, "technique instructions", skill["instruction"])
    result += entry_prompt("", "suggested uses", skill["domain"])
    result += entry_prompt("", "observed performance", skill["score"])
    return result


# TODO: move these prompt contents into format_prompt().
def skill_selection_prompt(job: dict, candidate_skills: list):
    result = ""
    result += (
        "We are working together on the job described below. "
        + "Our progress so far is described by the list of tasks below. "
        + "The first task describes the overall job. "
        + "Each successive task describes a technique we have chosen to follow "
        + "and our results so far. "
        + "The last task is our most recent technique.\n\n"
    )

    result += job_prompt("", job) + "\n\n"

    result += (
        "Now let's choose another technique technique to try. "
        + "To help guide us, we have a list of techniques that have been tried "
        + "before on similar jobs.  We can choose one of those techniques "
        + "or suggest a completely new technique to try.\n\n"
    )
    result += (
        "Below is our list of known techniques.  Each technique includes "
        + "the instructions for how to perform it, and a summary of our previous "
        + "experience with it.\n\n"
    )

    i = 1
    for skill in candidate_skills:
        result += skill_prompt(section_prompt("", i), skill)
        i += 1

    result += "That's all. " + "Please indicate what technique we should try next.\n\n"
    return result


def color_prompt(color: int, text: str):
    return f"\033[{color}m\033[1m{text}\033[0m\033[0m"


# This Tool can be used in a Chain using find_skills(),
# or in an Agent using __call__().
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
        return cls(vectorstore=vectorstore, docstore=docstore)

    @classmethod
    def create_example(cls, **kwargs):
        """Construct an example skill database with 4 example skills."""
        result = SkillStore.create()
        for skill in example_skills():
            result.add_skill(skill)
        return result

    def add_skill(self, skill: dict):
        """Insert one new skills into the vectorstore."""
        self.docstore.append(skill)
        docid = len(self.docstore) - 1
        skill_doc = skill_prompt("", skill)
        self.vectorstore.add_texts(texts=[skill_doc], metadatas=[{"id": docid}])

    def find_skills(self, job: dict):
        """Lookup candidate skills in the vectorstore."""
        # compose a query string from fields of solver_context.
        # focus on the most immediate task.
        task = job["tasks"][-1]
        query = task_prompt("", task)
        docs = self.vectorstore.similarity_search_with_score(query, k=4)
        result = []
        for doc in docs:
            docid = doc[0].metadata["id"]
            skill = self.docstore[docid]
            result.append(self.docstore[docid])
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
    llm: Any  # BaseLLM = Field(init=False)
    skilldb: SkillStore = Field(init=False)

    @classmethod
    def create(cls, skilldb: SkillStore, llm: BaseLLM) -> Chain:
        return cls(llm=llm, skilldb=skilldb)

    @property
    def input_keys(self) -> List[str]:
        """Acepts the whole problem context."""
        return ["job"]

    @property
    def output_keys(self) -> List[str]:
        """Acepts the whole problem context."""
        return ["selected_skill"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Lookup candidates in the vectorstore, and filter the candidates
        using the selection_prompt."""

        job = inputs["job"]
        skills = self.skilldb.find_skills(job)
        prompt_str = skill_selection_prompt(job, skills)
        prompt_value = PromptTemplate(
            template=prompt_str, input_variables=[]
        ).format_prompt()
        print(color_prompt(92, "hhh: skill_finder_prompt:"))
        print(prompt_value.to_string())
        llm_result = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
        reply = ""
        for g in llm_result.generations[0]:
            reply += g.text
        return {"selected_skill": reply}
        # TODO: find the selected skill in the list of skills.


"""
Below is a JSON5 schema for the "Skill" and "Job" records:

{
  $id: "Skill",
  properties: {
    tool: {
      type: "string",
      description: "the tool to invoke, such as an LLM"
    }
    instruction: {
      type: "natural language",
      description: "the instructions to the tool, such as a prompt for an LLM"
    }
    domain: {
      type: "natural language",
      description: "suggested problem domains for the skill"
    }
    score: {
      type: "natural language",
      description: "observed past performance of the skill, including run counts"
    }
}

{
  $id: "Entry"
  properties: {
    name: {type: "string"},
    value: {type: "string"}
  }
}

{
  $id: "Task",
  properties: {
    skill: {$ref: "Skill"},
    parameters: {type: "array" items: {$ref: "Entry"}},
    results: {type: "array" items: {$ref: "Entry"}}
}

{
  $id: "Job"
  properties: {
    tasks: {type: "array" items: {$ref: "Task"}}
  }
}
"""


def test_poem():
    """Just excercises SkillStore.add_skill and SkillFinderChain.run
    The Claude LLM answers with a technique for writng a poem, starting with a concrete theme.
    """
    skilldb = SkillStore.create()
    fake_skill = {
        "instruction": "write a poem",
        "domain": "poetry",
        "score": "none",
    }
    skilldb.add_skill(fake_skill)

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

    llm = ChatAnthropic()
    chain = SkillFinderChain.create(skilldb=skilldb, llm=llm)
    reply = chain.run({"job": fake_job})

    print(color_prompt(95, "hhh: skill_finder_reply:"))
    print(reply)


def example_skills():
    """A few example skills to test the SkillStore and the SkillFinder."""
    related_topics_skill = {
        "instruction": ("Please select about 20 topics related to the topics below."),
        "domain": "selecting topics for creative works",
        "score": "this technique has been tried 151 times, and helped 21 times",
    }

    personal_topics_skill = {
        "instruction": (
            "Please filter the list of topics below according to the users interests. "
            + "Some of the users main interests are also listed below."
        ),
        "domain": "selecting topics for creative works",
        "score": "this technique has been tried 43 times, and helped 2 times",
    }

    timely_topics_skill = {
        "instruction": (
            "Please filter the list of topics below according to recent public engagement. "
            + "consider using a search engine to ascertain the level of public engagement in each. "
        ),
        "domain": "selecting topics for creative works",
        "score": "this technique has been tried 4 times, and helped 0 times",
    }

    math_skill = {
        "instruction": ("""Try symmetry reductions."""),
        "domain": "exact analytical solutions of nonlinear partial differential equation.",
        "score": "this technique has been tried 0 times.",
    }
    return [
        related_topics_skill,
        personal_topics_skill,
        timely_topics_skill,
        math_skill,
    ]


def test_vlog():
    """Sets up a skill database with 4 skills, and a problem context with one overall
    task and one previous skill-driven task.  Calls SkillFinderChain.run.
    The Claude LLM answers by sugesting the "filter by user interests" technique.
    (and explains his fine choice.)
    """
    current_job_skill = {
        "instruction": (
            "The user creates vlog videos for publication on youtube. "
            "Our job is to select several possible topics for her next video."
        ),
        "domain": "",
        "score": "",
    }

    known_skills = example_skills()

    current_job_task = {
        "skill": current_job_skill,
        "parameters": [],
        "results": [],
    }

    related_topics_task = {
        "skill": known_skills[0],
        "parameters": [
            {
                "name": "the existing topics",
                "value": (
                    """ "my new BMW - the good and the bad","""
                    + """ "hairstyles of the rich and famous","""
                    + """ "a cat's life"."""
                ),
            }
        ],
        "results": [
            {
                "name": "the related topics",
                "value": (
                    """ "known defects with Tesla autiopilot","""
                    + """ "EV's everywhere except America","""
                    + """ "Biden's infrastructure package and US highways","""
                    + """ "no fault auto insurance","""
                    + """ "hair style disasters","""
                    + """ "car styles of the next century","""
                    + """ "these cats are dressed in style","""
                    + """ "when a cat meets a lion","""
                    + """ "what's wrong with volkwagen","""
                    + """ "why not to be famous","""
                    + """ "the cat in the hat is canceled again","""
                    + """ "does a BMW cost more in maintanence?","""
                    + """ "glamour in a post-pandemic world","""
                    + """ "how much should you be investung in your hair?","""
                    + """ "all the cats cast in Keanu"."""
                ),
            }
        ],
    }

    current_job = {
        "tasks": [
            current_job_task,
            related_topics_task,
        ]
    }

    skilldb = SkillStore.create()
    for skill in known_skills:
        skilldb.add_skill(skill)
    llm = ChatAnthropic()
    chain = SkillFinderChain.create(skilldb=skilldb, llm=llm)
    reply = chain.run({"job": current_job})

    print(color_prompt(95, "hhh: skill_finder_reply:"))
    print(reply)


# test_poem()
test_vlog()
