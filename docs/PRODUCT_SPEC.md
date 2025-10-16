# Product Spec (Assignment UI)

## Intended User & Decision
- **User:** HR analyst or team lead in a mid-size company.
- **Decision supported:** Prioritize early, supportive outreach for employees at risk of heavy absenteeism (not hiring/firing/pay).
- **When used:** Monthly, after attendance data refresh.

## What the Tool Does
- Predicts whether an employee is likely to be a **“heavy absentee”** (top 25% of absence hours in a month).
- Shows: predicted **probability**, **decision threshold**, and a **short “why”** (top contributing inputs).

## What the Tool Does NOT Do (Limitations)
- Doesn’t determine cause, intent, or policy violations.
- Not to be used for disciplinary actions, hiring, firing, or pay decisions.
- Depends on historical patterns; may miss shifts due to new policies or events.

## Data & Model (from Assignment 2)
- **Model:** Regularized Logistic Regression with **calibrated probabilities**.
- **Preprocessing:** missing-value imputation, **one-hot** for categorical vars, **scaling** for numeric vars.
- **Label:** Heavy absentee = **top quartile** of monthly absence hours (balanced).
- **Decision threshold:** **0.48** (chosen on validation).
- **Fairness lens:** **Age** (protected group = **Age ≥ 40**).

## Fairness Mitigations We Will Communicate
1. Remove **Age** as an input feature.
2. **Reweight** training across (Age × Label) groups.
3. **Calibrate** probabilities (isotonic or similar) and use a **single global threshold**.

## Fairness Metrics We Will Display
- **SPD (Statistical Parity Difference)** → closer to 0 is fairer.
- **EOD (Equal Opportunity Difference)** → closer to 0 is fairer.
- **FPR difference** → closer to 0 is fairer.
We will show **Before vs After mitigation** side-by-side.

## Responsible Use Guidance (UI copy)
- Use predictions as **signals for supportive outreach**, not final decisions.
- Combine with context: manager notes, HR policy, employee consent.
- Re-evaluate performance and fairness each release.