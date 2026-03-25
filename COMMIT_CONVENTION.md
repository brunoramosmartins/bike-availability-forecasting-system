# Commit Convention

This project follows [Conventional Commits v1.0.0](https://www.conventionalcommits.org/).

## Format

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

## Types

| Type       | Description                                      | Example                                              |
|------------|--------------------------------------------------|------------------------------------------------------|
| `feat`     | New feature or capability                        | `feat(ingestion): add GBFS station_status fetcher`   |
| `fix`      | Bug fix                                          | `fix(ingestion): handle API timeout gracefully`      |
| `docs`     | Documentation only                               | `docs: add ROADMAP.md with full project phases`      |
| `style`    | Formatting, no code logic change                 | `style: fix linting warnings in pipeline module`     |
| `refactor` | Code restructuring, no behavior change           | `refactor(storage): extract DB connection to module` |
| `test`     | Adding or updating tests                         | `test(model): add unit tests for baseline predictor` |
| `chore`    | Tooling, config, CI, dependencies                | `chore: configure GitHub Actions for ingestion`      |
| `perf`     | Performance improvement                          | `perf(query): optimize station lookup with index`    |
| `ci`       | CI/CD configuration changes                      | `ci: add scheduled workflow for data collection`     |
| `build`    | Build system or dependency changes               | `build: add lightgbm to requirements.txt`            |

## Scopes

Scopes are optional but recommended. Use the module or component name:

| Scope          | Used for                              |
|----------------|---------------------------------------|
| `ingestion`    | Data collection pipeline              |
| `storage`      | Database schema, migrations, queries  |
| `dataset`      | Feature engineering, dataset building |
| `model`        | ML models, training, evaluation       |
| `monitoring`   | Drift detection, performance tracking |
| `api`          | FastAPI serving layer                 |
| `viz`          | Dashboards, visualization             |
| `config`       | Project configuration files           |

## Rules

1. **Subject line**: imperative mood, lowercase, no period, max 72 characters
2. **Body** (optional): explain *why*, not *what* — the diff shows *what*
3. **Breaking changes**: add `BREAKING CHANGE:` in the footer or `!` after the type
4. **Issue references**: use `Refs #123` or `Closes #123` in the footer

## Examples

```
feat(ingestion): add retry logic with exponential backoff

The GBFS API occasionally returns 503 during peak hours.
Added httpx retry with 3 attempts and exponential backoff.

Refs #12
```

```
chore: configure pre-commit hooks for linting

Adds ruff and black as pre-commit hooks to enforce
consistent code style across the project.
```

```
fix(storage): prevent duplicate records on concurrent inserts

Added ON CONFLICT clause to the upsert query using
(station_id, last_reported) as the unique constraint.

Closes #18
```
