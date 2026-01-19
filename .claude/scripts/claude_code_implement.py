#!/usr/bin/env python3
"""
Implement a GitHub issue using Claude Code CLI.

This script replaces the custom autonomous_agent.py with the official
Claude Code CLI tool, leveraging its built-in capabilities and staying
up-to-date with the latest features.
"""

import json
import subprocess
import sys

from loguru import logger


def get_issue_details(issue_number: int) -> dict:
    """
    Get issue details from GitHub.

    :param issue_number: Issue number to fetch
    :returns: Issue data dictionary
    """
    result = subprocess.run(
        ["gh", "issue", "view", str(issue_number), "--json", "title,body"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def implement_with_claude_code(issue_number: int, issue_data: dict) -> dict:
    """
    Use Claude Code CLI to implement an issue.

    :param issue_number: Issue number being implemented
    :param issue_data: Issue details from GitHub
    :returns: Result dictionary with success status
    """
    title = issue_data["title"]
    body = issue_data["body"]

    # Build the prompt for Claude Code CLI
    prompt = f"""Implement GitHub issue #{issue_number}.

TITLE: {title}

DESCRIPTION:
{body}

INSTRUCTIONS:
For CODE features, follow test-driven development:
1. Explore the project structure to understand the codebase
2. Write comprehensive unit tests for this feature
3. Implement the feature to make tests pass
4. Run the full test suite to ensure no regressions
5. If any tests fail, fix them iteratively

For DOCUMENTATION tasks (README, docs, guides):
1. Explore the project to understand what needs documenting
2. Create the documentation with clear, accurate content
3. Verify the file exists and contains the required sections
4. IGNORE any "Unit Tests Required" sections - documentation doesn't need programmatic tests

When you're done:
- All tests should pass (for code features)
- The implementation should be complete and ready to commit
- Provide a summary of what you implemented

IMPORTANT: Work autonomously to complete the task. Use all available tools to explore, implement, test, and verify your work."""

    logger.info(f"Starting Claude Code CLI for issue #{issue_number}")
    logger.debug(f"Prompt length: {len(prompt)} chars")

    try:
        # Run Claude Code CLI in print mode (non-interactive)
        result = subprocess.run(
            [
                "claude",
                "--print",  # Non-interactive mode
                "--output-format",
                "text",  # Simple text output
                "--max-budget-usd",
                "5.00",  # Safety limit
                "--no-session-persistence",  # Don't save session
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            check=False,  # Don't raise on non-zero exit
        )

        logger.info(f"Claude Code CLI completed with exit code: {result.returncode}")

        if result.returncode == 0:
            logger.info("Implementation successful")
            logger.debug(f"Output length: {len(result.stdout)} chars")

            return {
                "success": True,
                "summary": result.stdout.strip(),
                "exit_code": result.returncode,
            }
        else:
            logger.error(f"Implementation failed with exit code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")

            return {
                "success": False,
                "error": result.stderr,
                "output": result.stdout,
                "exit_code": result.returncode,
            }

    except subprocess.TimeoutExpired:
        logger.error("Claude Code CLI timed out after 10 minutes")
        return {
            "success": False,
            "error": "Timeout after 10 minutes",
            "exit_code": -1,
        }
    except Exception as e:
        logger.error(f"Unexpected error running Claude Code CLI: {e}")
        return {
            "success": False,
            "error": str(e),
            "exit_code": -1,
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: claude_code_implement.py <issue_number>")
        sys.exit(1)

    issue_number = int(sys.argv[1])

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info(f"Implementing issue #{issue_number} with Claude Code CLI")

    # Get issue details
    issue_data = get_issue_details(issue_number)
    logger.info(f"Issue title: {issue_data['title']}")

    # Implement using Claude Code CLI
    result = implement_with_claude_code(issue_number, issue_data)

    if result["success"]:
        print(json.dumps(result, indent=2))
        sys.exit(0)
    else:
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
