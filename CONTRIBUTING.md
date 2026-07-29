# How This Project Will Be Developed in Part B

This document outlines the development workflow for the ten-week Part B delivery of the Solar Scheduler Benchmarking project.

## Development Approach

- **Methodology:** Design Science Research (DSR) + Agile/Scrum hybrid
- **Sprint cadence:** Five two-week sprints
- **Sprint reviews:** At the end of each sprint, a working increment is demonstrated
- **Retrospectives:** Brief written retrospective after each sprint review

## Branching Strategy

```
main          — stable, deployable code at sprint boundaries
  ├── dev     — integration branch for active development
  ├── feature/forecasting-model
  ├── feature/rule-scheduler
  ├── feature/rl-scheduler
  ├── feature/benchmark-pipeline
  └── feature/dashboard
```

## Commit Convention

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `data:` — data processing or dataset updates

Example: `feat: implement random forest demand forecast with temporal features`

## Definition of Done

- [ ] Code runs without errors on the target dataset
- [ ] Unit tests pass (where applicable)
- [ ] Documentation updated (README, docstrings, architecture notes)
- [ ] Results logged in `notebooks/` with reproducible parameters
- [ ] GitHub Project board updated (card moved to Done)
- [ ] Sprint review notes added to `docs/sprint_reviews/`

## Data Handling Rules

1. **No live household data** — only public, de-identified or synthetic datasets
2. **All data sources documented** — origin, licence, preprocessing steps in `data/README.md`
3. **Synthetic data flagged** — clearly labelled when used in benchmarks

## Quality Gates

- Forecasting model must achieve R² ≥ 0.80 on held-out data before proceeding to scheduling layer
- Both schedulers must be evaluated on identical test conditions
- Every recommendation from both schedulers must be logged with inputs and model version
- Governance checklist must be reviewed at every sprint boundary

## Getting Involved

This is a solo Master's capstone project. External contributions (issues, suggestions) are welcome for post-capstone continuation, but the ten-week Part B scope is fixed and governed by the project charter in `docs/project_charter.md`.
