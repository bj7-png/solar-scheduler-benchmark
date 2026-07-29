# SolarScheduler-Benchmark — Fair Benchmarking of ML vs Rule-Based Scheduling for Australian Household Solar Self-Consumption

![Status](https://img.shields.io/badge/status-Part_A_complete-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11-blue)

> **Capstone Project:** Master of Information Technology, PROF909 Information Technology Project Part A  
> **Author:** Bijay Shrestha (SHEA25030) | **Supervisor:** Dr Babak Amiri  
> **Pathway:** Build | **Trimester:** 2, 2026

---

## The Problem

Australian households have installed more rooftop solar capacity per capita than almost any country in the world — 26.8 GW across 4.2 million homes by mid-2025 (Clean Energy Council, 2025a). Yet most owners still lack a simple, low-effort way to know when to use, store or export their electricity. The result: meaningful generation is wasted or exported at poor feed-in tariffs.

AI and machine learning are widely proposed as the answer, with simulation studies claiming reinforcement learning (RL) outperforms rule-based control by 8–47% (Zhang, Ma & Zhu, 2020). But the **only real-world field trial** found RL performed **25.5% worse** than simple rule-based control (Ruddick et al., 2024). Meanwhile, governance, privacy and equity questions are almost never addressed alongside technical accuracy.

This project asks an honest question: *does ML-based scheduling actually help Australian households, or are we building on simulation optimism?*

## The Evidence

- **Generation forecasting:** Ensemble ML (random forest, SVM, neural nets) reaches R² ≈ 0.95 on weather data, but no study validates household-level models on Australian data (Kalra, Rajput & Verma, 2024; Gaboitaolelwe et al., 2023).
- **Demand forecasting:** LSTM/GRU architectures dominate household load forecasting, yet Australian smart meter data remains untested (Mathumitha, Rathika & Manimala, 2024; Shi, Xu & Li, 2017).
- **Scheduling gap:** Simulation studies show RL superiority, but real-world evidence contradicts this. Governance (NIST AI RMF Govern/Map functions) is absent from 100% of technical papers reviewed (NIST, 2023; Pillitteri, 2014).

*See the full evidence review and gap analysis in [`docs/proposal.md`](docs/proposal.md).*

## The Solution

A **four-layer benchmarking prototype** that compares a transparent rule-based scheduler against a learned ML/RL scheduler on identical Australian data, with governance and privacy built in from day one:

```
┌─────────────────────────────────────────────────────────────┐
│  DASHBOARD — Forecast, recommendations, benchmark results   │
├─────────────────────────────────────────────────────────────┤
│  EVALUATION & GOVERNANCE LAYER                              │
│  • Self-consumption rate, grid import, cost metrics         │
│  • Fairness and accuracy logging                            │
│  • NIST AI RMF governance checklist                         │
├─────────────────────────────────────────────────────────────┤
│  DECISION LAYER — Two schedulers, identical test conditions │
│  • Rule-Based: transparent threshold logic                  │
│  • Learned: RL or supervised ML policy                      │
├─────────────────────────────────────────────────────────────┤
│  FORECASTING LAYER — Random Forest generation + demand      │
├─────────────────────────────────────────────────────────────┤
│  DATA LAYER — Ausgrid, AEMO public data + synthetic profiles│
└─────────────────────────────────────────────────────────────┘
```

### What distinguishes this project

1. **Honest benchmarking** — negative results are valid findings, not failures.
2. **Explainable by design** — random forest over black-box deep nets; every recommendation is traceable.
3. **Governance-integrated** — privacy, ethics and accountability are first-order requirements, not afterthoughts.
4. **Open and reproducible** — all data, code, parameters and decisions are versioned and public.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Data processing | pandas, numpy |
| Forecasting | scikit-learn (RandomForestRegressor) |
| RL / ML scheduling | stable-baselines3 (PPO) or custom Q-learning |
| Dashboard | Streamlit |
| Version control | Git + GitHub |
| Notebooks | Jupyter (exploration & reporting) |
| Environment | venv / conda |

## Setup

```bash
# Clone the repository
git clone https://github.com/bijayshrestha/solar-scheduler-benchmark.git
cd solar-scheduler-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download public datasets (manual step — see data/README.md)
# 1. Ausgrid Solar Home Electricity Dataset
# 2. AEMO public rooftop PV data

# Run the prototype pipeline
python src/pipeline.py

# Launch the dashboard
streamlit run src/dashboard.py
```

## Project Plan

📋 **[View the GitHub Projects board](https://github.com/bijayshrestha/solar-scheduler-benchmark/projects)** — populated backlog with Sprint 1 ready and Part B milestones tagged.

### Key Part B Milestones

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| M1 | 2 | Dataset cleaned + governance checklist approved |
| M2 | 5 | Forecasting model (R² ≥ 0.80) + rule-based scheduler working |
| M3 | 8 | Head-to-head benchmark completed |
| M4 | 10 | Final demo, evaluation report, portfolio handover |

*Detailed Gantt chart with critical path and dependencies: see [`docs/gantt_chart.png`](docs/gantt_chart.png).*

## Ethics, Privacy and Responsible AI

This project is bound by:
- **ACS Code of Professional Conduct** — Primacy of the Public Interest, Honesty
- **NIST AI Risk Management Framework (AI RMF 1.0)** — Govern, Map, Measure, Manage
- **Australian Privacy Principles (APP 1, 3, 5, 6, 11)** — Data minimisation, consent, security
- **Australia's AI Ethics Principles** — Transparency, fairness, accountability
- **EU AI Act (2024)** — Forward-looking benchmark for consumer-facing transparency
- **ISO/IEC 42001:2023** — AI management system alignment

**Key ethical dilemmas addressed by design:**
1. **Privacy vs optimisation:** Granular household data reveals daily routines. Mitigation: use only de-identified, public or synthetic data; live occupied-home data is explicitly out of scope.
2. **Accountability:** If a scheduler recommendation costs a household money, who is responsible? Mitigation: prototype is decision-support (human-in-the-loop), not autonomous control; every recommendation is logged with inputs, model version and traceability.

*Full ethics and governance framework: [`docs/ethics_governance.md`](docs/ethics_governance.md).*

## Sustainability

**Environmental:** Better solar self-consumption reduces grid reliance and fossil-fuel generation, directly supporting Australia's emissions reduction goals (Murugesan, 2008 — Green IT principles).

**Social equity:** AI optimisation only benefits households that already own solar, batteries and smart meters. Renters and lower-income households — with lower battery uptake (Clean Energy Council, 2025b) — are excluded. The evaluation report names this limitation explicitly.

## Documents

| Document | Location |
|----------|----------|
| Full Proposal (PDF) | [`docs/PROF909_A3_Proposal_SHEA25030_Shrestha.pdf`](docs/PROF909_A3_Proposal_SHEA25030_Shrestha.pdf) |
| Updated Project Charter | [`docs/project_charter.md`](docs/project_charter.md) |
| Risk Register | [`docs/risk_register.md`](docs/risk_register.md) |
| RACI Matrix | [`docs/raci_matrix.md`](docs/raci_matrix.md) |
| Methodology Document | [`docs/methodology.md`](docs/methodology.md) |
| Architecture Diagram | [`docs/architecture_diagram.md`](docs/architecture_diagram.md) |
| AI Use Statement | [`docs/ai_use_statement.md`](docs/ai_use_statement.md) |
| Gantt Chart | [`docs/gantt_chart.png`](docs/gantt_chart.png) |

## Contact

- **Bijay Shrestha** — [LinkedIn](https://linkedin.com/in/bijayshrestha) | bijay.shrestha@student.edu.au
- **Academic Supervisor:** Dr Babak Amiri — bamiri@institute.edu.au
- **Industry Mentor:** TBC

## Acknowledgements

- Dr Babak Amiri — methodological guidance and assessment supervision.
- Assessment 1 marker and A2 industry panel — feedback that sharpened the evidence base, scope and ethical framing.
- Generative AI tools (ChatGPT, Claude) — used for drafting assistance and document structuring; all claims verified against primary sources. See [`docs/ai_use_statement.md`](docs/ai_use_statement.md).
