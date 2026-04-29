import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# CONFIG
# =============================================================================
FIXED_CSV = r"IGS_L_V3\benchmark_fixed.csv"
SEVERITY_CSV = r"IGS_L_V3\benchmark_severity.csv"

FIG_DIR = "figures_v3_2"
TAB_DIR = "tables_v3_2"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

ANOMALY_ORDER = [
    "gate_deletion_1q",
    "gate_deletion_2q",
    "gate_insertion",
    "gate_substitution",
    "gate_reorder",
    "trojan_NOT",
    "trojan_H",
    "qubit_swap",
]

ANOMALY_LABELS = {
    "gate_deletion_1q": "Single-Qubit Gate Deletion",
    "gate_deletion_2q": "Two-Qubit Gate Deletion",
    "gate_insertion": "Gate Insertion",
    "gate_substitution": "Gate Substitution",
    "gate_reorder": "Gate Reordering",
    "trojan_NOT": "Trojan NOT",
    "trojan_H": "Trojan H",
    "qubit_swap": "Qubit Swap",
}

SIS_HIGH_THRESHOLD = 0.95
IGS_DETECTION_THRESHOLD = 0.95
OIS_DETECTION_THRESHOLD = 0.95

plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "figure.titlesize": 34,
})

# =============================================================================
# LOAD DATA
# =============================================================================
df_fixed = pd.read_csv(FIXED_CSV)
df_sev = pd.read_csv(SEVERITY_CSV)

df_fixed = df_fixed[df_fixed["anomaly"] != "none"].copy()
df_sev = df_sev[df_sev["anomaly"] != "none"].copy()

df_sev["severity"] = pd.to_numeric(df_sev["severity"], errors="coerce")

df_fixed_valid = df_fixed.dropna(subset=["SIS", "IGSL", "OIS_sim"]).copy()
df_sev_valid = df_sev.dropna(subset=["SIS", "IGSL", "OIS_sim", "severity"]).copy()

sev_levels = sorted(df_sev_valid["severity"].dropna().unique())

# =============================================================================
# HELPERS
# =============================================================================
def pretty_anomaly(name):
    return ANOMALY_LABELS.get(str(name), str(name))

def savefig(name, dpi=300):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")

def savecsv(df, name):
    path = os.path.join(TAB_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")

def order_anomalies(df):
    out = df.copy()
    if "anomaly" in out.columns:
        out["anomaly"] = pd.Categorical(out["anomaly"], categories=ANOMALY_ORDER, ordered=True)
        out = out.sort_values("anomaly")
    return out

def prettify_anomaly_column(df):
    out = df.copy()
    if "anomaly" in out.columns:
        out["anomaly"] = out["anomaly"].map(pretty_anomaly)
    return out

# =============================================================================
# TABLE 1: MAIN FIXED SUMMARY
# =============================================================================
summary_rows = []

for anom in ANOMALY_ORDER:
    sub = df_fixed_valid[df_fixed_valid["anomaly"] == anom]
    if len(sub) == 0:
        continue

    summary_rows.append({
        "anomaly": pretty_anomaly(anom),
        "n": len(sub),
        "mean_SIS": sub["SIS"].mean(),
        "std_SIS": sub["SIS"].std(),
        "mean_IGS": sub["IGSL"].mean(),
        "std_IGS": sub["IGSL"].std(),
        "mean_OIS": sub["OIS_sim"].mean(),
        "std_OIS": sub["OIS_sim"].std(),
        "mean_IGS_time_sec": sub["IGSL_time_sec"].mean() if "IGSL_time_sec" in sub.columns else np.nan,
        "mean_OIS_time_sec": sub["OIS_time_sec"].mean() if "OIS_time_sec" in sub.columns else np.nan,
    })

summary_df = pd.DataFrame(summary_rows)
savecsv(summary_df, "table_main_metric_summary_fixed.csv")

# =============================================================================
# TABLE 2: STRUCTURAL BLIND-SPOT DETECTION BY SEVERITY
# =============================================================================
blind_sev = df_sev_valid[df_sev_valid["SIS"] >= SIS_HIGH_THRESHOLD].copy()
blind_sev["IGS_detected"] = blind_sev["IGSL"] < IGS_DETECTION_THRESHOLD
blind_sev["OIS_detected"] = blind_sev["OIS_sim"] < OIS_DETECTION_THRESHOLD

severity_blind_table = (
    blind_sev.groupby("severity")
    .agg(
        Blind_Cases=("severity", "size"),
        IGS_Detected=("IGS_detected", "sum"),
        OIS_Detected=("OIS_detected", "sum"),
    )
    .reset_index()
)

severity_blind_table["IGS_Rate"] = (
    severity_blind_table["IGS_Detected"] / severity_blind_table["Blind_Cases"] * 100
)

severity_blind_table["OIS_Rate"] = (
    severity_blind_table["OIS_Detected"] / severity_blind_table["Blind_Cases"] * 100
)

total_row = pd.DataFrame([{
    "severity": "Total",
    "Blind_Cases": int(severity_blind_table["Blind_Cases"].sum()),
    "IGS_Detected": int(severity_blind_table["IGS_Detected"].sum()),
    "OIS_Detected": int(severity_blind_table["OIS_Detected"].sum()),
}])

total_row["IGS_Rate"] = total_row["IGS_Detected"] / total_row["Blind_Cases"] * 100
total_row["OIS_Rate"] = total_row["OIS_Detected"] / total_row["Blind_Cases"] * 100

severity_blind_table = pd.concat([severity_blind_table, total_row], ignore_index=True)

severity_blind_table["IGS / OIS (%)"] = severity_blind_table.apply(
    lambda row: f"{row['IGS_Rate']:.2f} / {row['OIS_Rate']:.2f}",
    axis=1
)

severity_blind_table_final = severity_blind_table[
    ["severity", "Blind_Cases", "IGS_Detected", "OIS_Detected", "IGS / OIS (%)"]
].rename(columns={
    "severity": "Severity",
    "Blind_Cases": "Blind Cases",
    "IGS_Detected": "IGS Detected",
    "OIS_Detected": "OIS Detected",
})

savecsv(severity_blind_table_final, "table_detection_performance_structural_blindspots_by_severity.csv")

# =============================================================================
# TABLE 3: IGS VS OIS CORRELATION BY SEVERITY
# =============================================================================
corr_rows = []

for sev in sev_levels:
    sub = df_sev_valid[df_sev_valid["severity"] == sev].copy()

    pearson_r, pearson_p = pearsonr(sub["IGSL"], sub["OIS_sim"])
    spearman_rho, spearman_p = spearmanr(sub["IGSL"], sub["OIS_sim"])

    corr_rows.append({
        "severity": sev,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "n": len(sub),
    })

corr_by_severity = pd.DataFrame(corr_rows)
savecsv(corr_by_severity, "table_igs_vs_ois_correlation_by_severity.csv")

# =============================================================================
# FIGURE 1: SIS BOXPLOT
# =============================================================================
def plot_faceted_boxplot(metric_col, ylabel, title, filename):
    fig, axes = plt.subplots(2, 4, figsize=(36, 18), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, anom in zip(axes, ANOMALY_ORDER):
        sub = df_sev_valid[df_sev_valid["anomaly"] == anom].copy()

        data = [
            sub[sub["severity"] == sev][metric_col].dropna().values
            for sev in sev_levels
        ]

        ax.boxplot(
            data,
            labels=[str(sev) for sev in sev_levels],
            showmeans=True,
            meanline=True,
            patch_artist=True,
            widths=0.70,
            boxprops=dict(linewidth=3, edgecolor="black", facecolor="lightgray"),
            whiskerprops=dict(linewidth=3, color="black"),
            capprops=dict(linewidth=3, color="black"),
            medianprops=dict(linewidth=3, color="black"),
            meanprops=dict(linewidth=3, color="black", linestyle="--"),
            flierprops=dict(marker="o", markersize=8, markeredgecolor="black", markerfacecolor="black")
        )

        ax.set_title(pretty_anomaly(anom), fontweight="bold")
        ax.set_ylim(-0.02, 1.05)

    for ax in axes[4:]:
        ax.set_xlabel("Severity Level")

    for ax in [axes[0], axes[4]]:
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontweight="bold", y=1.03)
    savefig(filename)

plot_faceted_boxplot(
    metric_col="SIS",
    ylabel="SIS Distribution",
    title="SIS Distribution Across Discrete Severity Levels",
    filename="Fig_severity_SIS_boxplot_faceted.png"
)

plot_faceted_boxplot(
    metric_col="IGSL",
    ylabel="IGS Distribution",
    title="IGS Distribution Across Discrete Severity Levels",
    filename="Fig_severity_IGS_boxplot_faceted.png"
)

plot_faceted_boxplot(
    metric_col="OIS_sim",
    ylabel="OIS Distribution",
    title="OIS Distribution Across Discrete Severity Levels",
    filename="Fig_severity_OIS_boxplot_faceted.png"
)

# =============================================================================
# FIGURE 4: THREE-PANEL SENSITIVITY COMPARISON
# =============================================================================
sis_bar = df_sev_valid.groupby(["anomaly", "severity"])["SIS"].mean().reset_index()
igs_bar = df_sev_valid.groupby(["anomaly", "severity"])["IGSL"].mean().reset_index()
ois_bar = df_sev_valid.groupby(["anomaly", "severity"])["OIS_sim"].mean().reset_index()

pivot_sis = sis_bar.pivot(index="anomaly", columns="severity", values="SIS").reindex(ANOMALY_ORDER)
pivot_igs = igs_bar.pivot(index="anomaly", columns="severity", values="IGSL").reindex(ANOMALY_ORDER)
pivot_ois = ois_bar.pivot(index="anomaly", columns="severity", values="OIS_sim").reindex(ANOMALY_ORDER)

x = np.arange(len(ANOMALY_ORDER))
width = 0.24

fig, axes = plt.subplots(1, len(sev_levels), figsize=(16 * len(sev_levels), 12), sharey=True)

for ax, sev in zip(axes, sev_levels):
    ax.bar(x - width, pivot_sis[sev], width=width, label="SIS")
    ax.bar(x, pivot_igs[sev], width=width, label="IGS")
    ax.bar(x + width, pivot_ois[sev], width=width, label="OIS")

    ax.set_title(f"Severity {sev}")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_anomaly(a) for a in ANOMALY_ORDER], rotation=30, ha="right")
    ax.set_xlabel("Anomaly Type")
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Mean Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("SIS, IGS, and OIS Sensitivity by Severity", y=1.10)

savefig("Fig_sensitivity_by_severity_3panel.png")

# =============================================================================
# FIGURE 5: THREE-PANEL CORRELATION
# =============================================================================
fig, axes = plt.subplots(1, len(sev_levels), figsize=(16 * len(sev_levels), 12), sharex=True, sharey=True)

legend_handles = None
legend_labels = None

for ax, sev in zip(axes, sev_levels):
    sub = df_sev_valid[df_sev_valid["severity"] == sev].copy()

    pearson_r, _ = pearsonr(sub["IGSL"], sub["OIS_sim"])
    spearman_rho, _ = spearmanr(sub["IGSL"], sub["OIS_sim"])

    for anom in ANOMALY_ORDER:
        ss = sub[sub["anomaly"] == anom]
        if len(ss) > 0:
            ax.scatter(
                ss["IGSL"],
                ss["OIS_sim"],
                s=80,
                alpha=0.65,
                label=pretty_anomaly(anom)
            )

    x_vals = sub["IGSL"].to_numpy()
    y_vals = sub["OIS_sim"].to_numpy()

    m, b = np.polyfit(x_vals, y_vals, 1)
    xline = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax.plot(xline, m * xline + b, linewidth=3)

    ax.set_title(f"Severity {sev}\nPearson r = {pearson_r:.3f}, Spearman ρ = {spearman_rho:.3f}")
    ax.set_xlabel("IGS")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

axes[0].set_ylabel("OIS")

if legend_handles is not None:
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13))

fig.suptitle("IGS and OIS Relationship Across Severity Levels", y=1.20)
savefig("Fig_IGS_vs_OIS_correlation_3panel.png")

# =============================================================================
# FIGURE 6: THREE-PANEL RUNTIME
# =============================================================================
if "IGSL_time_sec" in df_sev_valid.columns and "OIS_time_sec" in df_sev_valid.columns:
    fig, axes = plt.subplots(1, len(sev_levels), figsize=(16 * len(sev_levels), 12), sharey=True)

    for ax, sev in zip(axes, sev_levels):
        sub = df_sev_valid[df_sev_valid["severity"] == sev]

        igs_rt = (
            sub.groupby("qubits")
            .agg(mean_time=("IGSL_time_sec", "mean"), std_time=("IGSL_time_sec", "std"))
            .reset_index()
        )

        ois_rt = (
            sub.groupby("qubits")
            .agg(mean_time=("OIS_time_sec", "mean"), std_time=("OIS_time_sec", "std"))
            .reset_index()
        )

        ax.errorbar(
            igs_rt["qubits"],
            igs_rt["mean_time"],
            yerr=igs_rt["std_time"].fillna(0),
            marker="o",
            linewidth=3,
            markersize=10,
            capsize=6,
            label="IGS"
        )

        ax.errorbar(
            ois_rt["qubits"],
            ois_rt["mean_time"],
            yerr=ois_rt["std_time"].fillna(0),
            marker="s",
            linewidth=3,
            markersize=10,
            capsize=6,
            label="OIS"
        )

        ax.set_title(f"Severity {sev}")
        ax.set_xlabel("Qubit Count")

    axes[0].set_ylabel("Runtime (seconds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Runtime Comparison of IGS and OIS Across Severity Levels", y=1.10)

    savefig("Fig_runtime_3panel_by_severity.png")

# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
print("\n==================== KEY RESULTS ====================")
print(f"Total fixed anomalous samples: {len(df_fixed_valid)}")
print(f"Total severity samples: {len(df_sev_valid)}")
print(f"Severity blind-spot cases, SIS >= {SIS_HIGH_THRESHOLD}: {len(blind_sev)}")
print("\nDetection Performance in Structural Blind-Spots by Severity:")
print(severity_blind_table_final.to_string(index=False))
print("\nTables written to:", TAB_DIR)
print("Figures written to:", FIG_DIR)
print("====================================================")