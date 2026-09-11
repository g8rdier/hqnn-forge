# Contributing to hqnn-forge

Thanks for your interest in contributing. This document summarizes the workflow we follow so
changes stay easy to review and the history stays easy to read.

## Language

All project communication must be in **English** — README, docs, commit messages, PR
titles/descriptions, code comments, issue titles and comments. Keeping everything in one
language ensures the history is readable by any contributor or reviewer, regardless of
background.

## Guiding Principle: Keep it Clean and Simple

Reduce complexity and increase clarity. A clean `main` branch, understandable commit history,
and consistent processes lead to better software.

## Git Workflow

### 1. Issue First

Every change should start with a GitHub issue before a branch is opened. Issues provide context
for why a change is being made, create a reference point for discussion, and ensure the work is
intentional.

*   **Title Format:** Issue titles use the same conventional-commit prefixes as branches and
    commits (e.g., `feat:`, `fix:`, `docs:`).
*   **Description Structure:** Provide clear context, steps to reproduce (for bugs), and
    expected outcomes (for features).
*   **Exception:** Trivial fixes (typos, obvious broken links) may skip the issue and go
    directly to a branch and PR.
*   **Order doesn't matter:** Issues can be worked, edited, or commented on in whatever order
    the work actually requires — there's no need to process them in creation order. The only
    real constraint is the reverse: don't reference an issue number in a commit, PR, or another
    issue before that issue exists.

### 2. Branch Naming

Feature branches are named `<type>/<short-description>`, using the same type prefix as the
conventional commit that will result from the PR (e.g. `feat/user-auth`, `fix/login-bug`,
`docs/api-guide`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### 3. No Direct Commits to Main

All changes go through a feature branch and a pull request — never commit directly to `main`.
This keeps the history reviewable and associates every change with a PR number.

### 4. Pull Requests

*   **One PR per feature branch**, addressing a single, specific purpose.
*   **PR title** follows the same conventional-commit format as the resulting squash-merge
    commit (e.g. `feat: add user authentication`).
*   **Test plan required** for any PR that changes code behaviour: a markdown checklist of
    steps to verify the change works, checked off before merging. Pure documentation or
    configuration changes don't need one.

### 5. Squash Merge

PRs targeting `main` are squash-merged, condensing the branch's history into one commit whose
title includes both the issue number and the PR number (e.g. `feat: add user authentication
(fixes #12) (#42)`).

Experiment/research branches are the exception — use a regular merge there so the trail of what
was tried and why isn't lost to squashing.

### 6. Review

Most PRs are self-reviewed before merging: does the code work as intended, does it follow
project conventions, are there obvious security issues, is test coverage adequate, is
documentation updated. Changes that modify interfaces or break existing contracts benefit from
review by affected stakeholders before merging.

### 7. Hotfixes

Critical production issues use a `hotfix/` branch. An issue is still recommended (can be
created retroactively) and a PR is still required, but review can be expedited with a
stakeholder's immediate approval. Direct commits to `main` remain forbidden.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Commit titles
are entirely lowercase, prefixed with a type: `feat`, `fix`, `docs`, `style`, `refactor`,
`test`, `chore`.

```
feat: add user authentication endpoint
```

Keep commit bodies to at most 3 bullet points. If you need more, the work is probably better
split into smaller, more atomic commits.

## Versioning

Releases follow [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`), tagged (e.g.
`v1.2.0`) on the `main` merge commit that encapsulates the release. `CHANGELOG.md` follows
[Keep a Changelog](https://keepachangelog.com/) and is updated as part of the release PR.

## Using AI Coding Assistants

If you use an AI coding assistant to help write a contribution, that's fine on your feature
branch — but review and understand everything before opening the PR, the same as you would for
your own code: check it follows project conventions, look for security issues, confirm tests
are adequate and passing. You are responsible for what you submit regardless of how it was
produced. Commits should carry human authorship only — please don't include AI co-authorship
trailers (e.g. `Co-Authored-By: <assistant>`) or tool-generated session links in commits, PR
descriptions, or comments in this repository.
