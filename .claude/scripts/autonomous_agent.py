#!/usr/bin/env python3
"""
Autonomous agent using Anthropic API with tool use.

This agent can read files, write files, execute commands, and interact with
the filesystem to autonomously implement GitHub issues.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from anthropic import Anthropic
from loguru import logger


class AutonomousAgent:
    """Agent that can autonomously execute tasks using Claude with tool use."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize agent.

        :param api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = Anthropic(api_key=self.api_key)
        self.tools = self._define_tools()

    def _define_tools(self) -> list[dict]:
        """
        Define tools available to the agent.

        :returns: List of tool definitions for Claude API
        """
        return [
            {
                "name": "read_file",
                "description": "Read the contents of a file. Returns the file contents as a string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (relative or absolute)",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "read_multiple_files",
                "description": "Read multiple files at once. More efficient than calling read_file multiple times. Returns a dictionary mapping file paths to their contents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to read (relative or absolute)",
                        }
                    },
                    "required": ["paths"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file. Creates parent directories if needed. Returns success message.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "list_files",
                "description": "List files in a directory matching a pattern. Returns list of file paths.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory to search in (default: current directory)",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match (e.g., '*.py', '**/*.js')",
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "execute_command",
                "description": "Execute a shell command and return stdout, stderr, and exit code. Use for running tests, git commands, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                        "working_dir": {
                            "type": "string",
                            "description": "Working directory (default: current directory)",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "create_github_issue",
                "description": "Create a GitHub issue with optional labels and parent. Returns the issue number.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Issue title",
                        },
                        "body": {
                            "type": "string",
                            "description": "Issue body (markdown formatted)",
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels to apply to the issue (e.g., ['backend', 'enhancement'])",
                        },
                        "parent": {
                            "type": "integer",
                            "description": "Parent issue number (makes this a child issue)",
                        },
                    },
                    "required": ["title", "body"],
                },
            },
            {
                "name": "mark_complete",
                "description": "Mark the task as complete. Use this when you have successfully implemented the feature and all tests pass. Provide a brief summary of what was accomplished.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of what was implemented and tested",
                        }
                    },
                    "required": ["summary"],
                },
            },
        ]

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Execute a tool and return the result.

        :param tool_name: Name of tool to execute
        :param tool_input: Input parameters for the tool
        :returns: Tool execution result
        """
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        if tool_name == "read_file":
            return self._read_file(tool_input["path"])

        elif tool_name == "read_multiple_files":
            return self._read_multiple_files(tool_input["paths"])

        elif tool_name == "write_file":
            return self._write_file(tool_input["path"], tool_input["content"])

        elif tool_name == "list_files":
            directory = tool_input.get("directory", ".")
            pattern = tool_input["pattern"]
            return self._list_files(directory, pattern)

        elif tool_name == "execute_command":
            command = tool_input["command"]
            working_dir = tool_input.get("working_dir", ".")
            return self._execute_command(command, working_dir)

        elif tool_name == "create_github_issue":
            labels = tool_input.get("labels", [])
            parent = tool_input.get("parent")
            return self._create_github_issue(
                tool_input["title"], tool_input["body"], labels, parent
            )

        elif tool_name == "mark_complete":
            return {"_completion_signal": True, "summary": tool_input["summary"]}

        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            with open(path) as f:
                content = f.read()
            logger.debug(f"Read {len(content)} chars from {path}")
            # Truncate very large files
            if len(content) > 10000:
                return (
                        content[:10000] + f"\n... [truncated, total {len(content)} chars]"
                )
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def _read_multiple_files(self, paths: list[str]) -> dict:
        """Read multiple files at once and return as dictionary."""
        results = {}
        total_chars = 0

        for path in paths:
            try:
                with open(path) as f:
                    content = f.read()

                # Truncate individual files if needed
                if len(content) > 10000:
                    results[path] = (
                            content[:10000]
                            + f"\n... [truncated, total {len(content)} chars]"
                    )
                    total_chars += 10000
                else:
                    results[path] = content
                    total_chars += len(content)

            except Exception as e:
                results[path] = f"Error reading file: {e}"

        logger.debug(f"Read {len(paths)} files, {total_chars} total chars")
        return results

    def _write_file(self, path: str, content: str) -> str:
        """Write content to file."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Wrote {len(content)} chars to {path}")
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _list_files(self, directory: str, pattern: str) -> str:
        """List files matching pattern."""
        try:
            from glob import glob

            dir_path = Path(directory)
            matches = glob(str(dir_path / pattern), recursive=True)
            # Filter out hidden and build directories
            matches = [
                m
                for m in matches
                if not any(part.startswith(".") for part in Path(m).parts)
                   and "node_modules" not in m
                   and "__pycache__" not in m
            ]
            return "\n".join(sorted(matches))
        except Exception as e:
            return f"Error listing files: {e}"

    def _execute_command(self, command: str, working_dir: str) -> str:
        """Execute shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            output = f"Exit code: {result.returncode}\n\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            logger.debug(f"Command output: {output[:500]}...")
            # Truncate very large command outputs
            if len(output) > 5000:
                return output[:5000] + f"\n... [truncated, total {len(output)} chars]"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 5 minutes"
        except Exception as e:
            return f"Error executing command: {e}"

    def _create_github_issue(
            self, title: str, body: str, labels: list[str] = None, parent: int = None
    ) -> str:
        """Create GitHub issue using gh CLI."""
        try:
            cmd = ["gh", "issue", "create", "--title", title, "--body", body]

            # Add labels if provided
            if labels:
                for label in labels:
                    cmd.extend(["--label", label])

            # Add parent issue if provided (creates child issue)
            if parent:
                # Use GitHub's GraphQL API to set parent relationship
                # First create the issue, then update it with parent
                pass  # Will be handled below

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            # Extract issue number from URL
            url = result.stdout.strip()
            issue_num = url.split("/")[-1]

            # Add parent relationship if specified (use GraphQL API)
            if parent:
                try:
                    # Get repository owner and name
                    repo_result = subprocess.run(
                        ["gh", "repo", "view", "--json", "owner,name"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    import json

                    repo_info = json.loads(repo_result.stdout)
                    owner = repo_info["owner"]["login"]
                    name = repo_info["name"]

                    # Get parent issue node ID
                    parent_query = f'query {{ repository(owner: "{owner}", name: "{name}") {{ issue(number: {parent}) {{ id }} }} }}'
                    parent_result = subprocess.run(
                        [
                            "gh",
                            "api",
                            "graphql",
                            "--jq",
                            ".data.repository.issue.id",
                            "-f",
                            f"query={parent_query}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    parent_id = parent_result.stdout.strip()

                    # Get child issue node ID
                    child_query = f'query {{ repository(owner: "{owner}", name: "{name}") {{ issue(number: {issue_num}) {{ id }} }} }}'
                    child_result = subprocess.run(
                        [
                            "gh",
                            "api",
                            "graphql",
                            "--jq",
                            ".data.repository.issue.id",
                            "-f",
                            f"query={child_query}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    child_id = child_result.stdout.strip()

                    # Add sub-issue relationship
                    mutation = f'mutation {{ addSubIssue(input: {{ issueId: "{parent_id}", subIssueId: "{child_id}" }}) {{ issue {{ id }} }} }}'
                    subprocess.run(
                        [
                            "gh",
                            "api",
                            "graphql",
                            "-H",
                            "GraphQL-Features: sub_issues",
                            "-f",
                            f"query={mutation}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    logger.info(f"Added issue #{issue_num} as child of #{parent}")
                except Exception as e:
                    logger.warning(f"Failed to set parent relationship: {e}")
                    # Don't fail the whole operation if parent setting fails

            labels_str = f" (labels: {', '.join(labels)})" if labels else ""
            parent_str = f" (child of #{parent})" if parent else ""
            logger.info(f"Created issue #{issue_num}: {title}{labels_str}{parent_str}")
            return f"Created issue #{issue_num}: {url}{parent_str}"
        except Exception as e:
            return f"Error creating issue: {e}"

    def run(self, task: str, max_turns: int = 30) -> dict:
        """
        Run agent on a task with autonomous tool use.

        :param task: Task description for the agent
        :param max_turns: Maximum number of conversation turns
        :returns: Dictionary with result and metadata including usage stats
        """
        logger.info(f"Starting agent task (max {max_turns} turns)")
        logger.debug(f"Task: {task[:200]}...")

        # System prompt with caching
        system_prompt = [
            {
                "type": "text",
                "text": """You are an autonomous coding agent that can read files, write files, execute commands, and create GitHub issues.

Work efficiently and complete tasks as quickly as possible. When you have successfully completed the task (e.g., all tests pass), use the mark_complete tool immediately to signal completion.

Do not continue working after the task is complete. Do not re-read files or re-run tests unnecessarily.""",
                "cache_control": {"type": "ephemeral"},
            }
        ]

        messages = [{"role": "user", "content": task}]
        total_input_tokens = 0
        total_output_tokens = 0

        for turn in range(max_turns):
            logger.info(f"Turn {turn + 1}/{max_turns}")

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=messages,
                system=system_prompt,
            )

            # Track token usage
            if hasattr(response, "usage"):
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens
                logger.debug(
                    f"Turn usage: {response.usage.input_tokens} in, "
                    f"{response.usage.output_tokens} out"
                )

            logger.debug(f"Response stop_reason: {response.stop_reason}")

            # Add assistant's response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Check if we're done
            if response.stop_reason == "end_turn":
                # Extract final text response
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text

                # Calculate cost (Sonnet 4.5 pricing: $3/M input, $15/M output)
                cost = (total_input_tokens / 1_000_000 * 3.0) + (
                        total_output_tokens / 1_000_000 * 15.0
                )

                logger.info(
                    f"Agent completed task - Tokens: {total_input_tokens} in, "
                    f"{total_output_tokens} out, Cost: ${cost:.4f}"
                )
                return {
                    "success": True,
                    "result": final_text,
                    "turns": turn + 1,
                    "usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens,
                        "estimated_cost": round(cost, 4),
                    },
                }

            # Process tool uses
            if response.stop_reason == "tool_use":
                tool_results = []
                completion_signaled = False

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        # Execute the tool
                        result = self._execute_tool(tool_name, tool_input)

                        # Check for completion signal
                        if isinstance(result, dict) and result.get(
                                "_completion_signal"
                        ):
                            completion_signaled = True
                            completion_summary = result.get("summary", "Task completed")

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": str(result),
                            }
                        )

                # Check if completion was signaled
                if completion_signaled:
                    cost = (total_input_tokens / 1_000_000 * 3.0) + (
                            total_output_tokens / 1_000_000 * 15.0
                    )
                    logger.info(
                        f"Agent marked task complete - Tokens: {total_input_tokens} in, "
                        f"{total_output_tokens} out, Cost: ${cost:.4f}"
                    )
                    return {
                        "success": True,
                        "result": completion_summary,
                        "turns": turn + 1,
                        "usage": {
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "total_tokens": total_input_tokens + total_output_tokens,
                            "estimated_cost": round(cost, 4),
                        },
                    }

                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})

                # Prune message history to keep context manageable
                # Keep first message (original task) + ensure last messages form valid pairs
                if len(messages) > 12:
                    # Keep first message + last N messages, ensuring we start with assistant message
                    # This prevents breaking tool_use/tool_result pairs
                    keep_count = 11
                    trimmed = messages[-keep_count:]

                    # If first message in trimmed section is user (tool_result),
                    # we need to include the prior assistant message (tool_use)
                    if trimmed[0]["role"] == "user":
                        keep_count += 1
                        trimmed = messages[-keep_count:]

                    messages = [messages[0]] + trimmed
                    logger.debug(
                        f"Pruned message history, now {len(messages)} messages"
                    )

            else:
                logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
                break

        # Calculate cost even on max turns
        cost = (total_input_tokens / 1_000_000 * 3.0) + (
                total_output_tokens / 1_000_000 * 15.0
        )

        logger.warning(
            f"Agent reached max turns ({max_turns}) - "
            f"Tokens: {total_input_tokens} in, {total_output_tokens} out, "
            f"Cost: ${cost:.4f}"
        )
        return {
            "success": False,
            "result": "Max turns reached without completion",
            "turns": max_turns,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "estimated_cost": round(cost, 4),
            },
        }


def decompose_specification(
        spec_path: str = "README.md", epic_number: int = None, max_turns: int = 30
) -> dict:
    """
    Decompose specification into GitHub issues.

    :param spec_path: Path to specification file
    :param epic_number: Epic issue number (all created issues become children)
    :param max_turns: Maximum agent turns
    :returns: Result dictionary
    """
    agent = AutonomousAgent()

    parent_instruction = (
        f"\n\nCRITICAL: You MUST set parent={epic_number} parameter in EVERY call to create_github_issue. This makes them child issues of the epic."
        if epic_number
        else ""
    )

    task = f"""Read the specification file at '{spec_path}' and decompose it into implementable GitHub issues.

For each issue:
1. Create a clear, specific title
2. Write detailed acceptance criteria
3. Specify what unit tests are required (for code features only - skip for documentation-only tasks)
4. Identify dependencies on other issues

Guidelines:
- Break down large features into smaller, testable units
- Besides tests, do not create functionality that is not specifically requested
- Create issue #0 for test framework setup if needed
- Ensure each issue can be implemented independently once deps are met
- Order issues logically (foundations before features)
- For documentation tasks (README, docs, etc.), focus on content requirements rather than unit tests
- Use "Depends on #N" in issue body to indicate dependencies{parent_instruction}

Use the create_github_issue tool to create each issue.{' When calling create_github_issue, you MUST include parent=' + str(epic_number) + ' in every call.' if epic_number else ''} After creating all issues, provide a summary showing:
- Total number of issues created
- Issue IDs and titles
- Dependency relationships"""

    return agent.run(task, max_turns=max_turns)


def implement_issue(issue_number: int, max_turns: int = 25) -> dict:
    """
    Autonomously implement a GitHub issue.

    :param issue_number: Issue number to implement
    :param max_turns: Maximum agent turns (default: 25)
    :returns: Result dictionary
    """
    agent = AutonomousAgent()

    # Get issue details
    result = subprocess.run(
        ["gh", "issue", "view", str(issue_number), "--json", "title,body"],
        capture_output=True,
        text=True,
        check=True,
    )
    issue_data = json.loads(result.stdout)

    task = f"""Implement GitHub issue #{issue_number}.

TITLE: {issue_data['title']}

BODY:
{issue_data['body']}

For CODE features, follow test-driven development:
1. Explore the project structure to understand the codebase
   - Use read_multiple_files to read several files at once (more efficient than multiple read_file calls)
   - Example: read_multiple_files(["src/main.py", "src/utils.py", "tests/test_main.py"])
2. Write comprehensive unit tests for this feature
3. Implement the feature to make tests pass
4. Run the full test suite to ensure no regressions
5. If any tests fail, fix them iteratively

For DOCUMENTATION tasks (README, docs, guides):
1. Explore the project to understand what needs documenting
   - Use read_multiple_files to batch-read relevant files in one turn
2. Create the documentation with clear, accurate content
3. Verify the file exists and contains the required sections
4. Do NOT write unit tests for documentation content
5. IGNORE any "Unit Tests Required" sections in the issue - documentation doesn't need programmatic tests

IMPORTANT: When done (tests pass for code, or content is complete for docs), immediately use the mark_complete tool with a summary of what you implemented. Do not continue working after completion."""

    return agent.run(task, max_turns=max_turns)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: autonomous_agent.py <command> [args]")
        print("Commands:")
        print(
            "  decompose <spec_file> <epic_number>  - Decompose spec into child issues"
        )
        print("  implement <issue_num>                - Implement an issue")
        sys.exit(1)

    command = sys.argv[1]

    if command == "decompose":
        if len(sys.argv) < 4:
            print("Error: Spec file and epic number required")
            print("Usage: autonomous_agent.py decompose <spec_file> <epic_number>")
            sys.exit(1)
        spec_file = sys.argv[2]
        epic_number = int(sys.argv[3])
        result = decompose_specification(spec_file, epic_number=epic_number)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)

    elif command == "implement":
        if len(sys.argv) < 3:
            print("Error: Issue number required")
            sys.exit(1)
        issue_num = int(sys.argv[2])
        result = implement_issue(issue_num)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
