# Updated Project Charter

*Revised from Assessment 1 in response to marker and panel feedback on scope clarity and governance depth.*

## Problem Statement

Australian households with rooftop solar often waste generation and pay more than necessary because they lack a practical, low-effort way to know when to use, store or export electricity. Existing AI/ML approaches to this problem remain largely untested on Australian residential data and conditions, particularly under variable weather and tariff structures, and rarely address governance or privacy.

## Value Proposition

An AI model that is openly benchmarked against a transparent rule-based baseline and has measurable, honestly reported impacts on the self-consumption of rooftop solar households, with evidence that can be used to support consumer-facing tools, guide energy retailers and battery fitters, and inform government sustainability programs.

## Scope

### In Scope
- Rooftop PV (with and without battery storage) for single dwellings connected to the grid
- A machine learning and reinforcement learning model based on local weather and household consumption data
- A fair, like-for-like comparison against a rule-based scheduling baseline
- Self-consumption rate (%), grid electricity use (kWh) and grid electricity cost ($) as outcome measures, tested on public and synthetic datasets over the ten-week capstone period

### Out of Scope
- Changes to large-scale network or grid engineering
- Commercial or industrial sites
- New hardware (inverters, batteries, sensors)
- Live deployment in occupied real homes

## Stakeholders

| Role | Name | Responsibility |
|------|------|---------------|
| Sponsor | Unit Convenor / industry mentor | Approves scope and evaluates outcomes |
| End user | Australian rooftop solar household or energy retailer | Ultimate beneficiary |
| Technical lead | Bijay Shrestha (student) | Designs, builds and evaluates the model |
| Academic supervisor | Dr Babak Amiri | Provides methodological guidance and assessment |

## Success Metrics (SMART)

1. Achieve a solar generation and demand forecast accuracy of R-squared of at least 0.80 within the ten-week project.
2. Complete a fair, documented benchmark of at least two approaches (rule-based and ML/RL) on at least one public dataset by Week 8.
3. Report the resulting change in simulated self-consumption rate transparently, whether positive, negative or negligible.
4. Produce a documented data governance and privacy checklist addressing the NIST AI RMF Govern and Map functions by project completion.

## Top Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| No real Australian household-level solar/weather dataset | Use AEMO and Ausgrid public data combined with clearly flagged comparable datasets |
| ML/RL model may not outperform the rule-based baseline (Ruddick et al., 2024) | Frame the project as an honest benchmark rather than assuming ML superiority |
| Household data privacy concerns | Use only de-identified, public or synthetic data governed by the NIST AI RMF checklist from the outset |
| Solo student workload exceeds 250–350 hour envelope | Track hours per sprint; reduce scope sprint by sprint rather than compress at the end |
| Industry mentor or dataset holder does not eventuate | Proceed with AEMO and Ausgrid public data as the primary source regardless |

## Budget and Resources

- **Time budget:** 250–350 hours over ten weeks (≈ 25–35 hours/week)
- **Compute:** Personal laptop + free cloud resources (Google Colab if GPU needed)
- **Data:** Free public datasets (Ausgrid, AEMO)
- **Software:** Open-source Python stack (pandas, scikit-learn, stable-baselines3, Streamlit)
- **Storage:** GitHub (free public repo)

## Approval

| Approver | Signature / Date |
|----------|-----------------|
| Student (Bijay Shrestha) | ________________ |
| Academic Supervisor (Dr Babak Amiri) | ________________ |
