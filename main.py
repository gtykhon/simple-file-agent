"""Simple File Agent — CLI entry point.

Usage:
    python main.py "write a Python function that reverses a string to utils.py"
    python main.py "read utils.py"
    python main.py "list ."
    python main.py --interactive
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from agent import AgentRunner, MultiModelClient

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)s  %(name)s  %(message)s",
)


def run_once(instruction: str, client: MultiModelClient) -> int:
    runner = AgentRunner(client)
    result = runner.run(instruction)
    print(result)
    return 0 if result.startswith("✅") else 1


def run_interactive(client: MultiModelClient):
    print("Simple File Agent  |  type 'exit' to quit\n")
    while True:
        try:
            instruction = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if instruction.lower() in ("exit", "quit", "q"):
            break
        if not instruction:
            continue
        run_once(instruction, client)


def main():
    parser = argparse.ArgumentParser(description="Simple File Agent")
    parser.add_argument(
        "instruction",
        nargs="?",
        help="Natural-language file instruction to execute",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive REPL",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        help="Ollama model name (default: llama3.1)",
    )
    parser.add_argument(
        "--claude-model",
        default=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        help="Anthropic Claude model name",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("AGENT_WORKSPACE", "."),
        help="Workspace root directory (agent cannot touch files outside this)",
    )
    args = parser.parse_args()

    # Set workspace before any file ops
    os.environ["AGENT_WORKSPACE"] = str(args.workspace)

    client = MultiModelClient(
        ollama_model=args.ollama_model,
        claude_model=args.claude_model,
    )

    if args.interactive:
        run_interactive(client)
    elif args.instruction:
        sys.exit(run_once(args.instruction, client))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
