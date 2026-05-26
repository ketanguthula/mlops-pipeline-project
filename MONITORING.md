# Model Monitoring and Drift Analysis

## Overview

This project uses Evidently AI to monitor feature drift between reference data and simulated production data. Drift monitoring helps identify when incoming data begins to differ significantly from the data used during model training.

The monitoring pipeline is implemented in:

```text
src/monitor_drift.py 