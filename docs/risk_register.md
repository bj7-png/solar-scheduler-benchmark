# Risk Register

*Likelihood and impact scored 1 (lowest) to 5 (highest); score = likelihood × impact. Reviewed at every sprint boundary.*

| ID | Risk | Likelihood | Impact | Score | Mitigation | Owner |
|:---|:---|:---:|:---:|:---:|:---|:---|
| R1 | No true Australian household-level solar and weather dataset is available | 3 | 5 | 15 | Use AEMO and Ausgrid public data, supplemented with clearly labelled comparable overseas datasets | Student |
| R2 | Learned scheduler fails to outperform the rule-based baseline | 3 | 4 | 12 | Frame the project as an honest benchmark; a negative result is still a valid, reportable finding | Student |
| R3 | Household consumption data raises privacy concerns | 2 | 5 | 10 | Use only de-identified, public or synthetic data, governed by the NIST AI RMF checklist | Student |
| R4 | Ten-week solo schedule slips due to data preparation delays | 3 | 3 | 9 | Identify two candidate datasets before Week 1; reduce scope sprint by sprint | Student |
| R5 | Reinforcement learning model fails to converge in the available time | 2 | 4 | 8 | Cap RL training time per sprint; fall back to a simpler supervised ML scheduler | Student |
| R6 | Forecasting model accuracy falls short of the R-squared 0.80 target | 2 | 3 | 6 | Use ensemble methods proven in evidence; retrain with additional features | Student |
| R7 | Solo student workload exceeds the 250 to 350 hour envelope | 2 | 3 | 6 | Track hours per sprint against the plan; escalate to supervisor if trending over | Student |
| R8 | Industry mentor or dataset holder introduction does not eventuate | 2 | 2 | 4 | Proceed with AEMO and Ausgrid public data as the primary source regardless of mentor access | Student |
| R9 | GitHub portfolio repository is incomplete or unclear for external (recruiter) audiences | 2 | 2 | 4 | Review README and board against the checklist at the end of every sprint | Student |
| R10 | Benchmark results are misinterpreted as proof that ML is universally superior or inferior | 1 | 3 | 3 | State findings as specific to the tested datasets and conditions in the evaluation report | Student |

## Risk Review Log

| Sprint | Date | Risks Reviewed | New Risks | Escalations |
|--------|------|---------------|-----------|-------------|
| Sprint 1 | TBD | R1, R3, R4, R8 | — | — |
| Sprint 2 | TBD | R1, R4, R6 | — | — |
| Sprint 3 | TBD | R2, R5, R6 | — | — |
| Sprint 4 | TBD | R2, R5, R7 | — | — |
| Sprint 5 | TBD | R7, R9, R10 | — | — |
