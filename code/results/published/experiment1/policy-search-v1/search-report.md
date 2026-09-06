# E1 Policy Search

Status: complete; rounds: 10/10.

Dev-only adaptive screening. No CAL/Q3/Q4 inference is run by this search.
Scores below are empirical; each scope has only 32 underlying Dev questions.

| Round | Level | Retention | Paired Mean | Worst Family | Action Rate | Reused |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | G0U0 | False | 62.5% | 62.5% | 62.5% | False |
| 1 | G0U1 | False | 0.0% | 0.0% | 81.2% | False |
| 1 | G1U0 | False | 41.4% | 0.0% | 41.4% | False |
| 1 | G1U1 | True | 0.0% | 0.0% | 81.2% | False |
| 2 | G0U0 | False | 71.9% | 71.9% | 71.9% | False |
| 2 | G0U1 | False | 0.0% | 0.0% | 81.2% | True |
| 2 | G1U0 | False | 74.2% | 62.5% | 82.0% | False |
| 2 | G1U1 | True | 0.0% | 0.0% | 81.2% | True |
| 3 | G0U0 | False | 12.5% | 12.5% | 18.8% | False |
| 3 | G0U1 | False | 0.0% | 0.0% | 81.2% | False |
| 3 | G1U0 | False | 74.2% | 62.5% | 82.0% | True |
| 3 | G1U1 | True | 0.0% | 0.0% | 81.2% | True |
| 4 | G0U0 | False | 62.5% | 62.5% | 62.5% | True |
| 4 | G0U1 | False | 0.0% | 0.0% | 81.2% | True |
| 4 | G1U0 | False | 29.7% | 3.1% | 29.7% | False |
| 4 | G1U1 | True | 0.0% | 0.0% | 81.2% | False |
| 5 | G0U0 | False | 90.6% | 90.6% | 90.6% | False |
| 5 | G0U1 | False | 0.0% | 0.0% | 81.2% | True |
| 5 | G1U0 | False | 59.4% | 37.5% | 73.4% | False |
| 5 | G1U1 | True | 0.0% | 0.0% | 81.2% | True |
| 6 | G0U0 | False | 9.4% | 9.4% | 9.4% | False |
| 6 | G0U1 | False | 0.0% | 0.0% | 81.2% | False |
| 6 | G1U0 | False | 74.2% | 62.5% | 82.0% | True |
| 6 | G1U1 | True | 0.0% | 0.0% | 81.2% | True |
| 7 | G0U0 | False | 62.5% | 62.5% | 62.5% | True |
| 7 | G0U1 | False | 0.0% | 0.0% | 81.2% | True |
| 7 | G1U0 | False | 32.0% | 0.0% | 32.0% | False |
| 7 | G1U1 | False | 0.0% | 0.0% | 81.2% | False |
| 8 | G0U0 | False | 0.0% | 0.0% | 0.0% | False |
| 8 | G0U1 | False | 0.0% | 0.0% | 81.2% | True |
| 8 | G1U0 | True | 73.4% | 50.0% | 73.4% | False |
| 8 | G1U1 | True | 0.0% | 0.0% | 81.2% | True |
| 9 | G0U0 | False | 100.0% | 100.0% | 100.0% | False |
| 9 | G0U1 | True | 0.0% | 0.0% | 81.2% | False |
| 9 | G1U0 | False | 74.2% | 62.5% | 82.0% | True |
| 9 | G1U1 | True | 0.0% | 0.0% | 81.2% | True |
| 10 | G0U0 | False | 0.0% | 0.0% | 0.0% | True |
| 10 | G0U1 | False | 0.0% | 0.0% | 81.2% | True |
| 10 | G1U0 | True | 51.6% | 15.6% | 51.6% | False |
| 10 | G1U1 | True | 0.0% | 0.0% | 81.2% | False |

Best round by level (not a declaration of research success):

- G0U0: round 6
- G0U1: round 9
- G1U0: round 8
- G1U1: round 1
