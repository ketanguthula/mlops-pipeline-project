# Drift Monitoring Analysis

## Overview

This project uses Evidently AI to monitor feature drift between reference training data and simulated production data. The monitoring pipeline is implemented in:

```text
src/monitor_drift.py
```

The script generates an HTML drift report at:

```text
reports/data_drift_report.html
```

The purpose of drift monitoring is to detect when incoming production data begins to differ significantly from the data distribution used during model training.

---

## Drifted Features

To simulate realistic production drift, the monitoring script intentionally modifies the following features:

- `MonthlyIncome`
- `Age`

`MonthlyIncome` is increased by 25% to simulate salary inflation or compensation changes over time. `Age` is increased by 3 years to simulate demographic shifts in the employee population.

During the most recent monitoring run, the system reported a drift share of approximately:

```text
0.06
```

This value is below the configured drift threshold of:

```text
0.30
```

The monitoring system therefore concluded that the observed drift remained within acceptable limits.

---

## Likely Impact on Model Performance

The detected drift is relatively small and is unlikely to significantly reduce model performance at this stage. However, continued changes in employee salary distributions or workforce demographics could eventually impact prediction quality.

Features such as `MonthlyIncome` and `Age` may influence employee attrition patterns. If these feature distributions continue changing over time, the relationships learned during model training may no longer accurately reflect production data.

As a result, model accuracy, recall, and F1 score could gradually degrade.

---

## Recommended Action

The recommended action at the current drift level is continued monitoring.

Because the detected drift remains below the configured threshold, immediate retraining is not necessary. However, if future monitoring runs show drift exceeding the threshold, the following actions are recommended:

1. Investigate which features are drifting most significantly
2. Re-evaluate model performance metrics
3. Retrain the model using newer production data
4. Reassess feature engineering and preprocessing steps

Continuous monitoring helps ensure that the model remains reliable and aligned with evolving production data.