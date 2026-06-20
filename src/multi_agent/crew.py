from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from multi_agent.evaluation import record_task_completion_callback
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    FinancialMetricsTool,
    GoogleSearchTool,
    MarketProfileTool,
    OfficialDisclosureSearchTool,
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

    def _settings(self) -> InvestmentResearchSettings:
        # 所有运行参数统一从 settings 中读取，避免散落在各个工具里。
        return InvestmentResearchSettings.from_env()

    def _artifact_path(self, name: str) -> str:
        settings = self._settings()
        return str(Path(settings.artifacts_dir) / name)

    def _local_pdf_path(self) -> Path | None:
        settings = self._settings()
        if not settings.local_filing_pdf_path.strip():
            return None

        candidate_path = Path(settings.local_filing_pdf_path)
        if not candidate_path.is_absolute():
            candidate_path = Path(__file__).resolve().parents[2] / candidate_path

        if candidate_path.exists():
            return candidate_path
        return None

    def _llm(self, model_name: str | None = None) -> LLM:
        # LLM 配置统一收束到这里，便于后续替换模型或调整采样参数。
        settings = self._settings()
        return LLM(
            model=model_name or settings.model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.3,
        )

    @agent
    def information_gathering_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["information_gathering_analyst"],  # type: ignore[index]
            llm=self._llm(
                self._settings().market_identifier_model
                or self._settings().company_resolver_model
                or self._settings().model
            ),
            tools=[
                MarketProfileTool(settings=self._settings()),
                GoogleSearchTool(settings=self._settings()),
                OfficialDisclosureSearchTool(settings=self._settings()),
                SecFilingSearchTool(settings=self._settings()),
            ],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def financial_statement_analyst(self) -> Agent:
        tools = [
            MarketProfileTool(settings=self._settings()),
            OfficialDisclosureSearchTool(settings=self._settings()),
            SecCompanyFactsTool(settings=self._settings()),
            FinancialMetricsTool(settings=self._settings()),
        ]
        if self._local_pdf_path() is not None:
            tools.insert(2, PDFTextExtractTool())

        return Agent(
            config=self.agents_config["financial_statement_analyst"],  # type: ignore[index]
            llm=self._llm(),
            tools=tools,
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def report_writing_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writing_analyst"],  # type: ignore[index]
            llm=self._llm(),
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
            output_file=self._settings().final_report_path,
            callback=record_task_completion_callback,
        )

    @crew
    def crew(self) -> Crew:
        """创建顺序执行的投研工作流。"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            cache=True,
            output_log_file=self._artifact_path("05_runtime"),
            verbose=True,
        )
