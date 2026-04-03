"""Finite State Machine engine for the agent workflow.

States flow: classify -> plan -> execute -> verify -> complete
All transitions are explicit — invalid outcomes raise ValueError (no silent fallback).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"complete", "error"}


@dataclass
class StateDefinition:
    """One node in the workflow graph."""
    name: str
    description: str = ""
    transitions: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """The full workflow graph, loaded from YAML."""
    name: str
    initial_state: str
    states: Dict[str, StateDefinition]

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowDefinition":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Workflow not found: {path}")
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        states = {}
        for name, defn in raw.get("states", {}).items():
            defn = defn or {}
            states[name] = StateDefinition(
                name=name,
                description=defn.get("description", ""),
                transitions=defn.get("transitions", {}),
            )

        initial = raw.get("initial_state", "classify")
        if initial not in states:
            raise ValueError(f"Initial state '{initial}' not in states: {list(states)}")

        return cls(name=raw.get("name", p.stem), initial_state=initial, states=states)


class AgentFSM:
    """
    Drives the agent through its workflow.

    Responsibilities:
      - Track current state
      - Validate and apply transitions
      - Detect terminal states
      - Record transition history

    Does NOT contain business logic.
    """

    def __init__(self, workflow: WorkflowDefinition):
        self.workflow = workflow
        self.current_state: str = workflow.initial_state
        self._history: List[Tuple[str, str, str]] = []  # (from, outcome, to)

    @property
    def state_def(self) -> StateDefinition:
        return self.workflow.states[self.current_state]

    def transition(self, outcome: str) -> str:
        """
        Move to the next state based on *outcome*.
        Raises ValueError for any unknown outcome — no silent fallback.
        Returns the new state name.
        """
        transitions = self.state_def.transitions
        if outcome not in transitions:
            raise ValueError(
                f"Invalid transition: state='{self.current_state}' outcome='{outcome}'. "
                f"Allowed outcomes: {list(transitions.keys())}"
            )
        new_state = transitions[outcome]
        self._history.append((self.current_state, outcome, new_state))
        logger.debug("FSM: %s --[%s]--> %s", self.current_state, outcome, new_state)
        self.current_state = new_state
        return new_state

    def is_terminal(self) -> bool:
        return self.current_state in TERMINAL_STATES

    @property
    def history(self) -> List[Tuple[str, str, str]]:
        return list(self._history)

    def reset(self):
        self.current_state = self.workflow.initial_state
        self._history.clear()
