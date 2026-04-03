"""Agent runner — drives the FSM through one instruction.

Flow:
  classify -> plan -> execute -> verify -> complete (or error at any step)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_ops import FILE_OPS, list_files, read_file, write_file
from .fsm import AgentFSM, WorkflowDefinition
from .llm_client import MultiModelClient
from .quality_gate import QualityGate

logger = logging.getLogger(__name__)

WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "code_operation.yaml"

# --------------------------------------------------------------------------- #
# System prompts                                                                #
# --------------------------------------------------------------------------- #

CLASSIFY_SYSTEM = """Classify the user instruction into exactly one operation type.
Reply with ONLY one word from: READ, WRITE, EDIT, DELETE, LIST, GENERATE_CODE"""

PLAN_SYSTEM = """You are a planning assistant. Analyse the instruction and return a JSON plan.
Reply with ONLY valid JSON, no markdown fences:
{
  "operation":          "read | write | edit | delete | list | generate_code",
  "path":               "<relative file path, or null>",
  "content":            "<full new content for write, or null>",
  "old_text":           "<exact text to replace for edit, or null>",
  "new_text":           "<replacement text for edit, or null>",
  "generation_prompt":  "<detailed prompt for code generation, or null>"
}"""

# --------------------------------------------------------------------------- #
# Context object                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class AgentContext:
    instruction: str
    operation_type: Optional[str] = None
    plan: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    model_log: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #

class AgentRunner:
    """
    Runs a single natural-language instruction through the full FSM pipeline.

    Args:
        client:      MultiModelClient for LLM calls.
        workflow:    WorkflowDefinition to drive the FSM.
        max_retries: Passed to QualityGate for code generation retries.
    """

    def __init__(
        self,
        client: MultiModelClient,
        workflow: Optional[WorkflowDefinition] = None,
        max_retries: int = 2,
    ):
        self.client = client
        self.workflow = workflow or WorkflowDefinition.from_yaml(str(WORKFLOW_PATH))
        self.quality_gate = QualityGate(client, max_retries=max_retries)

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def run(self, instruction: str) -> str:
        """
        Execute *instruction* and return a human-readable result string.
        Never raises — errors are caught and returned as an error message.
        """
        ctx = AgentContext(instruction=instruction)
        fsm = AgentFSM(self.workflow)

        try:
            self._classify(ctx, fsm)
            self._plan(ctx, fsm)
            self._execute(ctx, fsm)
            self._verify(ctx, fsm)
        except Exception as exc:
            ctx.error = str(exc)
            ctx.history.append(f"[ERROR] {exc}")
            logger.error("AgentRunner error: %s", exc, exc_info=True)
            # Force terminal state without going through FSM (exception path)
            fsm.current_state = "error"

        self._log_run(ctx, fsm)

        if fsm.current_state == "complete":
            return f"✅  {ctx.result}"
        return f"❌  Error: {ctx.error or 'unknown'}"

    # ---------------------------------------------------------------------- #
    # FSM state handlers                                                       #
    # ---------------------------------------------------------------------- #

    def _classify(self, ctx: AgentContext, fsm: AgentFSM):
        response, model = self.client.generate(ctx.instruction, system=CLASSIFY_SYSTEM)
        ctx.operation_type = response.strip().upper().split()[0]
        ctx.model_log.append(f"classify:{model}")
        ctx.history.append(f"[CLASSIFY] {ctx.operation_type}")
        fsm.transition("success")

    def _plan(self, ctx: AgentContext, fsm: AgentFSM):
        response, model = self.client.generate(
            f"Instruction: {ctx.instruction}",
            system=PLAN_SYSTEM,
            force_cloud=True,
        )
        plan = self._parse_json(response)
        if not plan:
            raise ValueError(f"Planner returned unparseable JSON:\n{response[:300]}")
        ctx.plan = plan
        ctx.model_log.append(f"plan:{model}")
        ctx.history.append(f"[PLAN] operation={plan.get('operation')} path={plan.get('path')}")
        fsm.transition("success")

    def _execute(self, ctx: AgentContext, fsm: AgentFSM):
        plan = ctx.plan
        op = (plan.get("operation") or "").lower()
        path = plan.get("path") or ""

        if op == "generate_code":
            gen_prompt = plan.get("generation_prompt") or ctx.instruction
            code, review = self.quality_gate.generate_and_review(
                gen_prompt, context=f"Target file: {path}"
            )
            if path:
                write_file(path, code)
                ctx.result = (
                    f"Generated and saved to '{path}' "
                    f"(quality score: {review.score:.2f}, approved: {review.approved})"
                )
            else:
                ctx.result = code
        elif op in FILE_OPS:
            op_fn = FILE_OPS[op]
            ctx.result = op_fn(
                path,
                content=plan.get("content", ""),
                old_text=plan.get("old_text", ""),
                new_text=plan.get("new_text", ""),
            )
        else:
            raise ValueError(f"Unknown operation: '{op}'")

        ctx.history.append(f"[EXECUTE] {op} -> {str(ctx.result)[:80]}")
        fsm.transition("success")

    def _verify(self, ctx: AgentContext, fsm: AgentFSM):
        """
        Lightweight verification: confirm mutating operations actually took effect.
        For read/list/generate_code (no path), trust the execute result.
        """
        plan = ctx.plan
        op = (plan.get("operation") or "").lower()
        path = plan.get("path") or ""

        verified = True

        if op in ("write", "edit", "generate_code") and path:
            try:
                read_file(path)
            except FileNotFoundError:
                verified = False

        elif op == "delete" and path:
            from pathlib import Path as _Path
            import os
            workspace = _Path(os.environ.get("AGENT_WORKSPACE", ".")).resolve()
            verified = not (workspace / path).exists()

        ctx.history.append(f"[VERIFY] {'OK' if verified else 'FAILED'}")

        if verified:
            fsm.transition("success")
        else:
            raise RuntimeError(f"Verification failed for '{op}' on '{path}'")

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        text = text.strip()
        for attempt in [text, re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()]:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _log_run(ctx: AgentContext, fsm: AgentFSM):
        logger.info(
            "Run complete | state=%s | op=%s | models=%s",
            fsm.current_state,
            ctx.operation_type,
            ", ".join(ctx.model_log),
        )
        for step in ctx.history:
            logger.debug("  %s", step)
