import os
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from multi_agent.core.model_routing import ModelRouter
from multi_agent.core.review_contracts import (
    validate_analysis_review_output,
    validate_report_review_output,
)
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

    def configure_run(
        self,
        *,
        model_tier_overrides: dict[str, str] | None = None,
        rerun_targets: list[str] | None = None,
    ) -> None:
        """Receive Flow-owned routing instructions without changing Crew topology."""
        self._model_tier_overrides = dict(model_tier_overrides or {})
        self._rerun_targets = list(rerun_targets or [])

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

    def _tier_for_agent(self, agent_name: str, default_tier: str) -> str:
        overrides = getattr(self, "_model_tier_overrides", {})
        return str(overrides.get(agent_name, default_tier))

    @agent
    def market_validation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_validation_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier(self._tier_for_agent("market_validation_analyst", "fast")),
            tools=[MarketValidationTool(settings=self._settings())],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def event_guidance_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["event_guidance_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier(self._tier_for_agent("event_guidance_analyst", "fast")),
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
            llm=self._llm_for_tier(self._tier_for_agent("fundamental_analyst", "deep")),
            tools=tools,
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def quant_valuation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["quant_valuation_analyst"],  # type: ignore[index]
            llm=self._llm_for_tier(self._tier_for_agent("quant_valuation_analyst", "deep")),
            tools=[FinancialMetricsTool(settings=self._settings())],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def data_quality_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_quality_reviewer"],  # type: ignore[index]
            llm=self._llm_for_tier(self._tier_for_agent("data_quality_reviewer", "review")),
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
            llm=self._llm_for_tier(self._tier_for_agent("report_writing_analyst", "deep")),
            tools=[],
            max_retry_limit=3,
            verbose=True,
        )

    @agent
    def logic_compliance_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["logic_compliance_reviewer"],  # type: ignore[index]
            llm=self._llm_for_tier(self._tier_for_agent("logic_compliance_reviewer", "review")),
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
            guardrail=validate_analysis_review_output,
            guardrail_max_retries=2,
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
            guardrail=validate_report_review_output,
            guardrail_max_retries=2,
            callback=record_task_completion_callback,
        )

    def _make_crew(self, *, tasks: list[Task]) -> Crew:
        self._ensure_crewai_storage_dir()
        return Crew(
            agents=self._all_agents(),
            tasks=tasks,
            process=Process.sequential,
            cache=True,
            output_log_file=self._artifact_path("05_runtime"),
            verbose=True,
        )

    def _all_agents(self) -> list[BaseAgent]:
        return [
            self.market_validation_analyst(),
            self.event_guidance_analyst(),
            self.fundamental_analyst(),
            self.quant_valuation_analyst(),
            self.data_quality_reviewer(),
            self.report_writing_analyst(),
            self.logic_compliance_reviewer(),
        ]

    def analysis_crew(self) -> Crew:
        """First five agents: research evidence and the strict data-quality contract."""
        return self._make_crew(
            tasks=[
                self.market_validation_task(),
                self.market_intelligence_task(),
                self.filing_review_task(),
                self.financial_analysis_task(),
                self.data_quality_review_task(),
            ]
        )

    def targeted_analysis_crew(self, targets: list[str]) -> Crew:
        """Rerun only the agent tasks named by typed RepairAction targets."""
        expanded_targets = list(dict.fromkeys(targets))
        if "fundamental_analyst" in expanded_targets:
            expanded_targets.append("quant_valuation_analyst")
        if {
            "market_validation_analyst",
            "event_guidance_analyst",
            "fundamental_analyst",
            "quant_valuation_analyst",
        }.intersection(expanded_targets):
            expanded_targets.append("data_quality_reviewer")
        target_order = (
            "market_validation_analyst",
            "event_guidance_analyst",
            "fundamental_analyst",
            "quant_valuation_analyst",
            "data_quality_reviewer",
        )
        expanded_target_set = set(expanded_targets)
        expanded_targets = [target for target in target_order if target in expanded_target_set]
        task_specs = {
            "market_validation_analyst": (
                "market_validation_task", self.market_validation_analyst,
                "00_market_validation.md",
            ),
            "event_guidance_analyst": (
                "market_intelligence_task", self.event_guidance_analyst,
                "01_market_intelligence.md",
            ),
            "fundamental_analyst": (
                "filing_review_task", self.fundamental_analyst,
                "02_filing_review.md",
            ),
            "quant_valuation_analyst": (
                "financial_analysis_task", self.quant_valuation_analyst,
                "03_financial_analysis.md",
            ),
            "data_quality_reviewer": (
                "data_quality_review_task", self.data_quality_reviewer,
                "08_data_quality_review.md",
            ),
        }
        context_targets = {
            "event_guidance_analyst": ("market_validation_analyst",),
            "fundamental_analyst": (
                "market_validation_analyst",
                "event_guidance_analyst",
            ),
            "quant_valuation_analyst": (
                "market_validation_analyst",
                "event_guidance_analyst",
                "fundamental_analyst",
            ),
            "data_quality_reviewer": (
                "market_validation_analyst",
                "event_guidance_analyst",
                "fundamental_analyst",
                "quant_valuation_analyst",
            ),
        }
        tasks_by_target: dict[str, Task] = {}
        for target in expanded_targets:
            spec = task_specs.get(target)
            if spec is None:
                continue
            task_name, agent_factory, output_name = spec
            context = [
                tasks_by_target[dependency]
                for dependency in context_targets.get(target, ())
                if dependency in tasks_by_target
            ]
            tasks_by_target[target] = Task(
                name=task_name,
                config=self.tasks_config[task_name],  # type: ignore[index]
                agent=agent_factory(),
                context=context,
                output_file=self._task_output_file(self._artifact_path(output_name)),
                guardrail=(
                    validate_analysis_review_output
                    if target == "data_quality_reviewer"
                    else None
                ),
                guardrail_max_retries=2,
                callback=record_task_completion_callback,
            )
        tasks = list(tasks_by_target.values())
        if not tasks:
            raise ValueError("targeted analysis requires at least one supported RepairAction target")
        return self._make_crew(tasks=tasks)

    def report_crew(self) -> Crew:
        """Run writer and logic reviewer only after Flow supplies REPORT_CONTEXT_JSON."""
        writer = Task(
            name="investment_report_task",
            config=self.tasks_config["investment_report_task"],  # type: ignore[index]
            agent=self.report_writing_analyst(),
            output_file=self._task_output_file(self._artifact_path("04_writer_payload.json")),
            callback=record_task_completion_callback,
        )
        reviewer = Task(
            name="logic_compliance_review_task",
            config=self.tasks_config["logic_compliance_review_task"],  # type: ignore[index]
            agent=self.logic_compliance_reviewer(),
            context=[writer],
            output_file=self._task_output_file(self._artifact_path("09_logic_compliance_review.md")),
            guardrail=validate_report_review_output,
            guardrail_max_retries=2,
            callback=record_task_completion_callback,
        )
        return self._make_crew(tasks=[writer, reviewer])

    @crew
    def crew(self) -> Crew:
        """创建顺序执行的投研工作流。"""
        return self._make_crew(
            tasks=[
                self.market_validation_task(),
                self.market_intelligence_task(),
                self.filing_review_task(),
                self.financial_analysis_task(),
                self.data_quality_review_task(),
                self.investment_report_task(),
                self.logic_compliance_review_task(),
            ]
        )
