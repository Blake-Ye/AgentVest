import os
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from multi_agent.core.model_routing import ModelRouter
from multi_agent.evaluation import record_task_completion_callback
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    FinancialMetricsTool,
    PDFTextExtractTool,
    SecCompanyFactsTool,
    SecFilingSearchTool,
)
from multi_agent.tools.market_validation import MarketValidationTool
from multi_agent.tools.review_tools import (
    CrossSourceConsistencyTool,
    EvidenceCoverageTool,
    FinancialFieldCompletenessTool,
    MarketToolPolicyAuditTool,
)
from multi_agent.tools.tavily_search import TavilySearchTool


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

    def _ensure_crewai_storage_dir(self) -> str:
        storage_dir = os.getenv("CREWAI_STORAGE_DIR", "").strip()
        if storage_dir:
            Path(storage_dir).mkdir(parents=True, exist_ok=True)
            return storage_dir

        settings = self._settings()
        default_storage_dir = Path(settings.artifact_root) / ".crewai_storage"
        default_storage_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CREWAI_STORAGE_DIR"] = str(default_storage_dir)
        return str(default_storage_dir)

    def _artifact_path(self, name: str) -> str:
        settings = self._settings()
        return str(Path(settings.artifacts_dir) / name)

    def _task_output_file(self, path_value: str | Path) -> str:
        target_path = Path(path_value)
        if not target_path.is_absolute():
            return str(target_path)

        project_root = Path(__file__).resolve().parents[2]
        try:
            return str(target_path.relative_to(project_root))
        except ValueError:
            return str(target_path)

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

    def _llm(self) -> LLM:
        return self._llm_for_tier("deep")

    def _llm_for_tier(self, tier: str) -> LLM:
        # ponytail: 先用简单的 tier -> model 映射接住 agent 分层；等 Flow 引入动态升级条件后再扩展。
        settings = self._settings()
        model = ModelRouter(settings).for_tier(tier)
        return LLM(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.3,
        )

    @agent
    def market_validation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_validation_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier("fast"),
            tools=[MarketValidationTool(settings=self._settings())],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def event_guidance_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["event_guidance_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier("fast"),
            tools=[
                TavilySearchTool(settings=self._settings()),
            ],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        tools = [
            SecFilingSearchTool(settings=self._settings()),
            SecCompanyFactsTool(settings=self._settings()),
        ]
        if self._local_pdf_path() is not None:
            tools.insert(2, PDFTextExtractTool())

        return Agent(
            config=self.agents_config["fundamental_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier("deep"),
            tools=tools,
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def quant_valuation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["quant_valuation_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier("deep"),
            tools=[FinancialMetricsTool(settings=self._settings())],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def data_quality_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_quality_reviewer"],  # type: ignore[index]
            llm=self._llm_for_tier("review"),
            tools=[
                EvidenceCoverageTool(),
                CrossSourceConsistencyTool(),
                MarketToolPolicyAuditTool(),
                FinancialFieldCompletenessTool(),
            ],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def report_writing_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writing_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier("deep"),
            tools=[],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def logic_compliance_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["logic_compliance_reviewer"],  # type: ignore[index]
            llm=self._llm_for_tier("review"),
            tools=[],
            max_retry_limit=3,
            verbose=True,
        )

    @task
    def market_validation_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_validation_task"],  # type: ignore[index]
            output_file=self._task_output_file(self._artifact_path("00_market_validation.md")),
            callback=record_task_completion_callback,
        )

    @task
    def market_intelligence_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_intelligence_task"],  # type: ignore[index]
            context=[self.market_validation_task()],
            output_file=self._task_output_file(self._artifact_path("01_market_intelligence.md")),
            callback=record_task_completion_callback,
        )

    @task
    def filing_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["filing_review_task"],  # type: ignore[index]
            context=[self.market_validation_task(), self.market_intelligence_task()],
            output_file=self._task_output_file(self._artifact_path("02_filing_review.md")),
            callback=record_task_completion_callback,
        )

    @task
    def financial_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["financial_analysis_task"],  # type: ignore[index]
            context=[self.market_validation_task(), self.filing_review_task()],
            output_file=self._task_output_file(self._artifact_path("03_financial_analysis.md")),
            callback=record_task_completion_callback,
        )

    @task
    def data_quality_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_quality_review_task"],  # type: ignore[index]
            context=[
                self.market_validation_task(),
                self.market_intelligence_task(),
                self.filing_review_task(),
                self.financial_analysis_task(),
            ],
            output_file=self._task_output_file(self._artifact_path("08_data_quality_review.md")),
            callback=record_task_completion_callback,
        )

    @task
    def investment_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["investment_report_task"],  # type: ignore[index]
            context=[
                self.market_validation_task(),
                self.market_intelligence_task(),
                self.filing_review_task(),
                self.financial_analysis_task(),
                self.data_quality_review_task(),
            ],
            output_file=self._task_output_file(self._settings().final_report_path),
            callback=record_task_completion_callback,
        )

    @task
    def logic_compliance_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["logic_compliance_review_task"],  # type: ignore[index]
            context=[self.investment_report_task(), self.data_quality_review_task()],
            output_file=self._task_output_file(self._artifact_path("09_logic_compliance_review.md")),
            callback=record_task_completion_callback,
        )

    @crew
    def crew(self) -> Crew:
        """创建顺序执行的投研工作流。"""
        self._ensure_crewai_storage_dir()
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            cache=True,
            output_log_file=self._artifact_path("05_runtime"),
            verbose=True,
        )
