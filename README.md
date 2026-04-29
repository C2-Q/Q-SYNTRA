# Q-SYNTRA
Q-SYNTRA (Quantum Structural Yield &amp; Neural Trajectory Reliability Analyzer) is an intelligent security framework designed for exascale quantum software.

# Multi-Level Integrity Evaluation of Quantum Circuits

This repository contains the replication package for our conference paper on evaluating quantum circuit integrity using structural, interaction-level, and behavioral perspectives.

## Overview

Quantum circuits in the NISQ era undergo multiple transformations before execution, including compilation, mapping, and hardware-specific optimizations. As a result, the same logical circuit can appear in different structural forms, and structural similarity does not always imply behavioral equivalence.

In this work, we study circuit integrity using three complementary metrics:

- **SIS (Structural Integrity Score)**  
  Measures global structural differences such as gate count, depth, and topology.

- **OIS (Operational Integrity Score)**  
  Measures behavioral deviation using Jensen–Shannon distance between output distributions.

- **IGS (Interaction Graph Score)**  
  Captures interaction patterns and dependencies between operations in a pre-execution setting.

The goal is to understand how these perspectives respond to controlled anomalies in quantum circuits.

---

## Repository Structure
.
├── src/
│ ├── Main_Metrics.py # Main benchmark code (SIS, OIS, IGS + anomaly injection)
│ └── analyze_results.py # Analysis and figure generation
│
├── data/
│ ├── benchmark_fixed.csv
│ ├── benchmark_severity.csv
│ └── benchmark_summary.csv
│
├── figures/
│ ├── Fig_severity_SIS_boxplot_faceted.png
│ ├── Fig_severity_IGS_boxplot_faceted.png
│ ├── Fig_severity_OIS_boxplot_faceted.png
│ ├── Fig_sensitivity_by_severity_3panel.png
│ ├── Fig_IGS_vs_OIS_correlation_3panel.png
│ └── Fig_runtime_3panel_by_severity.png
│
├── tables/
│ ├── table_main_metric_summary_fixed.csv
│ ├── table_detection_performance_structural_blindspots_by_severity.csv
│ └── table_igs_vs_ois_correlation_by_severity.csv
│
├── requirements.txt
└── README.md
