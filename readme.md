# Python PR Analyzer

A GitHub Actions–friendly tool that extracts code changes from Python pull requests and sends them to the [`auto-cepu`](https://github.com/mrrizal/auto-cepu) API for **LLM-powered automated code review**.

This project is intended to run automatically inside CI (GitHub Actions) and is configured entirely via environment variables (`.env`), with no CLI arguments.

---

## Features

- Parse GitHub PR diffs to extract added/deleted Python code and corresponding full function definitions.
- Send extracted code sections to the [`auto-cepu`](https://github.com/mrrizal/auto-cepu) API for analysis.
- Designed to run unattended in GitHub Actions.
- No manual arguments — fully configured via `.env`.

---

## How It Works

1. **GitHub Actions** triggers this script on pull request events.
2. It reads PR context and credentials from environment variables.
3. Fetches the PR diff from the GitHub API.
4. Extracts **only the changed Python code** (added/deleted lines along with their complete function definitions).
5. Sends the extracted code to your `auto-cepu` API endpoint.
6. `auto-cepu` responds with **LLM-based review feedback** (issues, suggestions, severity ratings, duplication analysis, etc.).
7. The analyzer automatically posts the review feedback as a reply to the pull request.
