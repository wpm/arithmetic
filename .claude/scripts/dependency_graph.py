#!/usr/bin/env python3
"""
Dependency graph operations for Epik project automation.

This script provides utilities for managing the dependency graph of GitHub issues,
including finding ready issues, validating the graph structure, and checking
project completion status.
"""

import sys
from enum import StrEnum
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class GraphIssue(BaseModel):
    """Represents a single issue in the dependency graph."""

    title: str = Field(..., description="Issue title")
    deps: list[int] = Field(
        default_factory=list, description="List of issue IDs this depends on"
    )

    @field_validator("deps")
    @classmethod
    def sort_deps(cls, v: list[int]) -> list[int]:
        """Sort dependencies for consistent output."""
        return sorted(v)


class DependencyGraph(BaseModel):
    """The complete dependency graph for a project run."""

    epic_label: Optional[str] = Field(None, description="Epic label scoping this graph")
    issues: dict[int, GraphIssue] = Field(
        ..., description="Map of issue ID to issue data"
    )

    def validate_structure(self) -> tuple[bool, Optional[str]]:
        """
        Validate graph structure for cycles and invalid references.

        :returns: Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        # Check all dependencies reference existing issues
        for issue_id, issue in self.issues.items():
            for dep in issue.deps:
                if dep not in self.issues:
                    return (
                        False,
                        f"Issue {issue_id} depends on non-existent issue {dep}",
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: int) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in self.issues[node].deps:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for issue_id in self.issues:
            if issue_id not in visited:
                if has_cycle(issue_id):
                    return False, "Cycle detected in dependency graph"

        return True, None

    def ready_issues(self, closed_issues: set[int]) -> list[int]:
        """
        Find issues ready to implement.

        An issue is ready if all its dependencies are closed and it is not closed itself.

        :param closed_issues: Set of issue IDs that are already closed
        :returns: Sorted list of issue IDs that are ready to implement
        """
        ready = []
        for issue_id, issue in self.issues.items():
            if issue_id in closed_issues:
                continue
            if all(dep in closed_issues for dep in issue.deps):
                ready.append(issue_id)
        return sorted(ready)

    def is_complete(self, closed_issues: set[int]) -> bool:
        """
        Check if all issues in the graph are closed.

        :param closed_issues: Set of issue IDs that are closed
        :returns: True if all issues are closed
        """
        return all(issue_id in closed_issues for issue_id in self.issues)


class Command(StrEnum):
    """Available commands for the dependency graph script."""

    READY = "ready"
    VALIDATE = "validate"
    COMPLETE = "complete"


app = typer.Typer(help="Dependency graph operations for one-shot automation")


def load_graph(graph_path: Path) -> DependencyGraph:
    """
    Load dependency graph from YAML file.

    :param graph_path: Path to the graph YAML file
    :returns: Parsed DependencyGraph object
    :raises: typer.Exit if file doesn't exist or is invalid
    """
    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        raise typer.Exit(1)

    try:
        with open(graph_path) as f:
            data = yaml.safe_load(f)
        return DependencyGraph(**data)
    except Exception as e:
        logger.error(f"Failed to parse graph file: {e}")
        raise typer.Exit(1)


def get_closed_issues(epic_label: Optional[str] = None) -> set[int]:
    """
    Get set of closed issue IDs from GitHub CLI.

    :param epic_label: Optional label to filter issues
    :returns: Set of closed issue IDs
    :raises: typer.Exit if gh command fails
    """
    import subprocess

    try:
        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "closed",
            "--json",
            "number",
            "--jq",
            ".[].number",
        ]

        # Add label filter if provided
        if epic_label:
            cmd.insert(3, "--label")
            cmd.insert(4, epic_label)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to query GitHub issues: {e}")
        raise typer.Exit(1)
    except FileNotFoundError:
        logger.error("GitHub CLI (gh) not found. Please install it.")
        raise typer.Exit(1)


@app.command()
def ready(
    graph_path: Path = typer.Option(
        ".claude/dependency-graph.yml",
        "--graph",
        "-g",
        help="Path to dependency graph YAML file",
    ),
) -> None:
    """
    Find issues ready to implement.

    An issue is ready when all its dependencies are closed and it is not closed itself.
    Outputs a space-separated list of issue IDs.
    """
    graph = load_graph(graph_path)
    closed = get_closed_issues(epic_label=graph.epic_label)
    ready_list = graph.ready_issues(closed)

    if ready_list:
        print(" ".join(map(str, ready_list)))
    else:
        logger.info("No ready issues found")


@app.command()
def validate(
    graph_path: Path = typer.Option(
        ".claude/dependency-graph.yml",
        "--graph",
        "-g",
        help="Path to dependency graph YAML file",
    ),
) -> None:
    """
    Validate dependency graph structure.

    Checks for cycles and invalid references. Exits with code 0 if valid, 1 if invalid.
    """
    graph = load_graph(graph_path)
    is_valid, error = graph.validate_structure()

    if is_valid:
        logger.info("Graph is valid")
    else:
        logger.error(f"Graph validation failed: {error}")
        raise typer.Exit(1)


@app.command()
def complete(
    graph_path: Path = typer.Option(
        ".claude/dependency-graph.yml",
        "--graph",
        "-g",
        help="Path to dependency graph YAML file",
    ),
) -> None:
    """
    Check if epic is complete.

    Returns exit code 0 if all issues are closed, 1 otherwise.
    """
    graph = load_graph(graph_path)
    closed = get_closed_issues(epic_label=graph.epic_label)

    epic_str = f"Epic '{graph.epic_label}'" if graph.epic_label else "Project"

    if graph.is_complete(closed):
        logger.info(f"{epic_str} complete: all issues closed")
    else:
        remaining = set(graph.issues.keys()) - closed
        logger.info(f"{epic_str} incomplete: {len(remaining)} issues remaining")
        logger.debug(f"Remaining issues: {sorted(remaining)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    app()
