from pathlib import Path
from functools import cached_property

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from multi_agent.evaluation import record_task_completion_callback
from multi_agent.runtime import prepare_runtime_env
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    FinancialMetricsTool,
    GoogleSearchTool,
    PDFTextExtractTool,
    SecCompanyFactsTool,
    SecFilingSearchTool,
)


@CrewBase
class MultiAgent:
    """工业级自动化投研 Crew。"""

    agents: list[BaseAgent]
    tasks: list[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @cached_property
    def settings(self) -> InvestmentResearchSettings:
        # 所有运行参数统一从 settings 中读取，避免散落在各个工具里。
        return InvestmentResearchSettings.from_env()

    def _artifact_path(self, name: str) -> str:
        return str(Path(self.settings.artifacts_dir) / name)

    def _local_pdf_path(self) -> Path | None:
        if not self.settings.local_filing_pdf_path.strip():
            return None

        candidate_path = Path(self.settings.local_filing_pdf_path)
        if not candidate_path.is_absolute():
            candidate_path = Path(__file__).resolve().parents[2] / candidate_path

        if candidate_path.exists():
            return candidate_path
        return None

    @cached_property
    def llm(self) -> LLM:
        # LLM 配置统一收束到这里，便于后续替换模型或调整采样参数。
        return LLM(
            model=self.settings.model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            temperature=0.3,
        )

    @cached_property
    def information_gathering_tools(self) -> list[object]:
        return [
            GoogleSearchTool(settings=self.settings),
            SecFilingSearchTool(settings=self.settings),
        ]

    @cached_property
    def financial_analysis_tools(self) -> list[object]:
        tools: list[object] = [
            SecCompanyFactsTool(settings=self.settings),
            FinancialMetricsTool(settings=self.settings),
        ]
        if self._local_pdf_path() is not None:
            tools.append(PDFTextExtractTool())
        return tools

    @agent
    def information_gathering_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["information_gathering_analyst"],  # type: ignore[index]
            llm=self.llm,
            tools=self.information_gathering_tools,
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def financial_statement_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_statement_analyst"],  # type: ignore[index]
            llm=self.llm,
            tools=self.financial_analysis_tools,
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def report_writing_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writing_analyst"],  # type: ignore[index]
            llm=self.llm,
            tools=[],
            max_retry_limit=3,
            verbose=True,
        )

    @task
    def market_intelligence_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_intelligence_task"],  # type: ignore[index]
            output_file=self._artifact_path("01_market_intelligence.md"),
            callback=record_task_completion_callback,
        )

    @task
    def filing_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["filing_review_task"],  # type: ignore[index]
            context=[self.market_intelligence_task()],
            output_file=self._artifact_path("02_filing_review.md"),
            callback=record_task_completion_callback,
        )

    @task
    def financial_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["financial_analysis_task"],  # type: ignore[index]
            context=[self.market_intelligence_task(), self.filing_review_task()],
            output_file=self._artifact_path("03_financial_analysis.md"),
            callback=record_task_completion_callback,
        )

    @task
    def investment_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["investment_report_task"],  # type: ignore[index]
            context=[
                self.market_intelligence_task(),
                self.filing_review_task(),
                self.financial_analysis_task(),
            ],
            output_file=self.settings.final_report_path,
            callback=record_task_completion_callback,
        )

    @crew
    def crew(self) -> Crew:
        """创建顺序执行的投研工作流。"""
        prepare_runtime_env()
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            cache=True,
            output_log_file=self._artifact_path("05_runtime"),
            verbose=True,
        )
