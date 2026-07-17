# Contributing to AI Face Mask Detection System

Thanks for considering a contribution! This document covers how the project is branched, the standards a pull request is expected to meet, and how to report bugs or propose features.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Set Up](#getting-set-up)
- [Git Branch Strategy](#git-branch-strategy)
- [Making a Change](#making-a-change)
- [Coding Standards](#coding-standards)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)

## Code of Conduct

Be respectful and constructive. Assume good intent, critique code/ideas rather than people, and keep discussion focused on making the project better. Harassment or discrimination of any kind won't be tolerated.

## Getting Set Up

Follow the [Installation](README.md#-installation) section of the README, using `requirements-dev.txt` (not just `requirements.txt`) so you have the linting/testing/formatting tools too:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
```

## Git Branch Strategy

- **`main`** - always stable/deployable, protected, merges only via reviewed PR with passing CI
- **`develop`** - integration branch where feature branches land before promoting to `main`
- **`feature/<name>`** - one branch per feature, e.g. `feature/multi-camera-support`, `feature/dark-mode`
- **`fix/<name>`** - targeted bug fixes, e.g. `fix/camera-disconnect-detection`
- **`release/vX.Y.Z`** - cut from `main` for versioned milestones

Branch off `develop` for anything that isn't a critical hotfix, and open your PR back against `develop`.

## Making a Change

1. Fork the repository and clone your fork.
2. Create a branch off `develop` following the naming convention above:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/my-change
   ```
3. Make your change, keeping it focused - separate, unrelated changes should be separate PRs.
4. Add or update tests for any behavior change (see [`tests/`](tests/) for existing patterns - hardware like the camera and the YOLO model are always mocked out, never touched directly, so tests stay fast and deterministic).
5. Update relevant docs (`README.md`, `docs/architecture.md`, `docs/performance.md`, or inline docstrings) if behavior, configuration, or setup steps changed.
6. Run the full check suite locally before opening a PR (see [Coding Standards](#coding-standards)).

## Coding Standards

This project enforces consistency via `pre-commit` and CI, not code review nitpicking:

```bash
pytest                        # unit tests
ruff check src tests          # linting
black --check src tests       # formatting
isort --check-only src tests  # import order
mypy src                      # static type checking
```

A few conventions to follow beyond what the tools enforce automatically:

- **Custom exceptions over bare `Exception`** - raise/catch the specific types in `src/mask_detector/utils/exceptions.py` wherever a failure is expected and recoverable (see [Error Handling](README.md#-error-handling)); reserve broad `except Exception` for true last-resort boundaries, and always log (`logger.exception(...)`) before showing a friendly message.
- **Modular, single-responsibility modules** - if you're adding a genuinely new concern (a new alert channel, a new storage backend, ...), give it its own module rather than growing an existing one indefinitely.
- **Docstrings explain *why*, not *what*** - comments and docstrings should capture non-obvious intent or trade-offs; avoid narrating what the next line of code obviously does.
- **No secrets or large binaries committed** - `.env`, model weights, logs, and `Screenshots/` are git-ignored on purpose; keep it that way.

## Commit Message Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>: <short summary>

[optional longer description]
```

Common `<type>` values: `feat` (new feature), `fix` (bug fix), `refactor`, `test`, `docs`, `chore`, `perf`. For example:

```
feat: add mask-type classification (surgical/N95/cloth)
fix: prevent screenshot cooldown from resetting on rerun
docs: document CAMERA_DISCONNECT_TIMEOUT_SECONDS in README
```

## Pull Request Process

1. Push your branch and open a PR against `develop` (not `main`).
2. Fill in what changed and why, and how you tested it (unit tests added/updated, manual webcam testing performed, etc.).
3. Ensure CI is green - the same `pytest`/`ruff`/`black`/`isort`/`mypy` checks from [Coding Standards](#coding-standards) run automatically on every PR via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
4. Address review feedback with new commits (no need to force-push/squash until asked).
5. Once approved and green, a maintainer will merge it into `develop`. `develop` is periodically promoted to `main` via a release PR.

## Reporting Bugs

Open a GitHub issue with:

- What you expected to happen vs. what actually happened
- Steps to reproduce (including relevant `.env` settings, OS, Python version, and camera/hardware if relevant)
- Any error message or traceback from the app or `logs/app.log`

Check the [Error Handling](README.md#-error-handling) table first - many failure modes already have a documented, expected behavior and fix.

## Proposing Features

Check the [Roadmap / Future Improvements](README.md#-roadmap--future-improvements) section first - your idea may already be tracked there. If not, open an issue describing:

- The problem it solves (not just the feature itself)
- Any relevant prior art (a library, API, or approach you'd suggest)
- Whether you're able/willing to implement it yourself
