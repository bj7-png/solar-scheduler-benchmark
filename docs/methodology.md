# Methodology and Delivery Approach

## Overarching Methodology: Design Science Research (DSR)

This project follows the Design Science Research methodology (Hevner et al., 2004; Hevner & Chatterjee, 2010), which fits build-style IT projects whose primary output is an artefact addressing an identified gap.

### Hevner's Three Cycles

| Cycle | How Satisfied | Evidence |
|-------|--------------|----------|
| **Relevance Cycle** | Engagement with the problem domain (Australian solar households) and existing evidence base (A1 review, AEMO data, industry reports) | A1 evidence review; AEMO (2025); Clean Energy Council (2025a, 2025b) |
| **Design Cycle** | Iterative build-evaluate of the benchmarking prototype across five sprints | Sprint increments: dataset → forecast model → rule scheduler → RL scheduler → benchmark |
| **Rigor Cycle** | Draws on forecasting literature (Kalra et al., 2024), scheduling literature (Ruddick et al., 2024), and governance frameworks (NIST AI RMF, APP, ACS Code) | 28 references; named frameworks applied to design decisions |

### Why DSR over alternatives

DSR is preferred over pure Agile because evaluation of the artefact against fairness metrics — rather than feature shipping — is the primary success criterion. DSR is preferred over a research-only methodology because the deliverable is a working prototype, not a study.

## Delivery Methodology: Agile/Scrum Hybrid

Within the DSR structure, delivery is organised as five two-week sprints following the Scrum framework (Schwaber & Sutherland, 2020), adapted for a solo practitioner.

### Sprint Structure

| Sprint | Weeks | Goal | Increment |
|--------|-------|------|-----------|
| Sprint 1 | 1–2 | Data & governance | Cleaned dataset + approved governance checklist |
| Sprint 2 | 3–4 | Forecasting model | Working RF generation/demand forecast (R² ≥ 0.80) |
| Sprint 3 | 5–6 | Rule-based baseline | Transparent rule scheduler + mid-project review |
| Sprint 4 | 7–8 | Learned scheduler + benchmark | RL/ML scheduler + head-to-head benchmark |
| Sprint 5 | 9–10 | Evaluation & handover | Final report, dashboard, portfolio close-out |

### Solo Scrum Adaptations

- **Daily stand-ups:** Written status updates in GitHub Issues (not verbal)
- **Sprint reviews:** Demonstration to academic supervisor at sprint boundary
- **Retrospectives:** Written reflection on what worked, what didn't, and adjustments for next sprint
- **Backlog:** GitHub Projects Kanban board with milestone labels

### Why Agile over Waterfall

The benchmark outcome — whether the learned scheduler beats the rule-based baseline — is genuinely uncertain (Ruddick et al., 2024 found the opposite of simulation optimism). An iterative approach lets scope be adjusted sprint by sprint if data preparation takes longer than planned, without risking the whole project on an assumption that the learned model will succeed.

## Quality Management

- **Version control:** Every artefact committed to GitHub with descriptive messages
- **Reproducibility:** All parameters, random seeds, and preprocessing steps documented
- **Evaluation protocol:** Identical test conditions for both schedulers; statistical confidence where sample size permits
- **Documentation:** Jupyter notebooks for exploration; markdown for architecture and decisions

## Ethical Governance Integration

Governance is not a separate phase — it is a cross-cutting concern reviewed at every sprint boundary:

- Sprint 1: Data governance checklist approved before any data is processed
- Sprint 2–4: Model cards and decision logs maintained for both schedulers
- Sprint 5: Final governance framework integrated into evaluation report
