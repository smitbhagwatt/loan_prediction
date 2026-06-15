# %% [markdown]
# # 📊 Exploratory Data Analysis — Loan Approval Prediction
#
# This notebook explores the loan application dataset to understand
# distributions, relationships, and patterns before building ML models.
#
# **Sections:**
# 1. Dataset Overview
# 2. Missing Value Analysis
# 3. Target Variable Distribution
# 4. Univariate Analysis
# 5. Bivariate Analysis (Feature vs Target)
# 6. Correlation Analysis
# 7. Key Insights

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Styling ─────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#e2e8f0",
    "axes.labelcolor": "#1e293b",
    "text.color": "#1e293b",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#e2e8f0",
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "font.size": 11,
})

PALETTE = ["#6366f1", "#22d3ee", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]
sns.set_palette(PALETTE)

# ── Load Data ───────────────────────────────
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_data.csv"
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ## 1. Dataset Overview

# %%
print("=" * 60)
print("DATASET INFO")
print("=" * 60)
print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nBasic statistics:")
df.describe(include="all").T

# %% [markdown]
# ## 2. Missing Value Analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Missing counts
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)

if len(missing) > 0:
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(missing)))
    missing.plot.barh(ax=axes[0], color=colors, edgecolor="#333")
    axes[0].set_title("Missing Values by Column", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Count")

    for i, (val, name) in enumerate(zip(missing.values, missing.index)):
        axes[0].text(val + 5, i, f"{val} ({val/len(df)*100:.1f}%)", va="center", fontsize=9)

    # Missing heatmap
    sns.heatmap(df.isnull().T, cbar=True, cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Missing Value Heatmap", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Row Index")
else:
    axes[0].text(0.5, 0.5, "No missing values!", ha="center", va="center", fontsize=14)
    axes[1].text(0.5, 0.5, "No missing values!", ha="center", va="center", fontsize=14)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_missing_values.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Target Variable Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
target_counts = df["Loan_Status"].value_counts()
colors_target = ["#10b981", "#ef4444"]
target_counts.plot.bar(ax=axes[0], color=colors_target, edgecolor="#333", width=0.5)
axes[0].set_title("Loan Status Distribution", fontweight="bold", fontsize=13)
axes[0].set_xlabel("Status (Y=Approved, N=Rejected)")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 20, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")

# Pie chart
axes[1].pie(target_counts.values, labels=["Approved (Y)", "Rejected (N)"],
            colors=colors_target, autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12}, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Approval Rate", fontweight="bold", fontsize=13)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_target_distribution.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Univariate Analysis

# %% [markdown]
# ### 4a. Categorical Features

# %%
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
            "Property_Area", "Credit_History"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    if i < len(axes):
        data = df[col].dropna().value_counts()
        data.plot.bar(ax=axes[i], color=PALETTE[:len(data)], edgecolor="#333", width=0.6)
        axes[i].set_title(col, fontweight="bold", fontsize=12)
        axes[i].set_ylabel("Count")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

# Hide unused axes
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Categorical Feature Distributions", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_categorical.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ### 4b. Numerical Features

# %%
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for i, col in enumerate(num_cols):
    # Histogram
    axes[0][i].hist(df[col].dropna(), bins=40, color=PALETTE[i], edgecolor="#333", alpha=0.8)
    axes[0][i].set_title(f"{col}\n(Distribution)", fontweight="bold", fontsize=11)
    axes[0][i].axvline(df[col].mean(), color="#ef4444", linestyle="--", label=f"Mean: {df[col].mean():.0f}")
    axes[0][i].axvline(df[col].median(), color="#22d3ee", linestyle="--", label=f"Median: {df[col].median():.0f}")
    axes[0][i].legend(fontsize=8)

    # Box plot
    axes[1][i].boxplot(df[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=PALETTE[i], alpha=0.6),
                       medianprops=dict(color="#ef4444", linewidth=2))
    axes[1][i].set_title(f"{col}\n(Box Plot)", fontweight="bold", fontsize=11)

plt.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_numerical.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Bivariate Analysis — Features vs Loan Approval

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

bivariate_cols = ["Gender", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area"]

for i, col in enumerate(bivariate_cols):
    ct = pd.crosstab(df[col], df["Loan_Status"], normalize="index") * 100
    ct.plot.bar(ax=axes[i], color=["#ef4444", "#10b981"], edgecolor="#333", width=0.6)
    axes[i].set_title(f"{col} vs Loan Approval", fontweight="bold", fontsize=12)
    axes[i].set_ylabel("Percentage (%)")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")
    axes[i].legend(["Rejected", "Approved"], fontsize=9)
    axes[i].set_ylim(0, 100)

plt.suptitle("Feature Impact on Loan Approval", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_bivariate.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ### Income Distribution by Loan Status

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, col in enumerate(["ApplicantIncome", "LoanAmount"]):
    for status, color, label in [("Y", "#10b981", "Approved"), ("N", "#ef4444", "Rejected")]:
        data = df[df["Loan_Status"] == status][col].dropna()
        axes[i].hist(data, bins=40, alpha=0.5, color=color, label=label, edgecolor="#333")
    axes[i].set_title(f"{col} by Loan Status", fontweight="bold", fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].legend()

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_income_vs_status.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Correlation Analysis

# %%
# Encode categoricals for correlation
df_encoded = df.copy()
for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed",
            "Property_Area", "Loan_Status"]:
    if col in df_encoded.columns:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes

# Drop ID
if "Loan_ID" in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=["Loan_ID"])

fig, ax = plt.subplots(figsize=(12, 9))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "outputs" / "eda_correlation.png",
            bbox_inches="tight", dpi=120)
plt.show()

# %% [markdown]
# ## 7. Key Insights
#
# Based on the exploratory analysis:
#
# 1. **Target Imbalance**: The dataset is moderately imbalanced — more approved than rejected loans
# 2. **Credit History**: The strongest predictor — applicants with good credit history are significantly more likely to be approved
# 3. **Income Skewness**: Both ApplicantIncome and LoanAmount are right-skewed → log transforms will help
# 4. **Missing Values**: Credit_History, Self_Employed, and LoanAmount have the most missing values
# 5. **Property Area**: Semiurban applicants have a slightly higher approval rate
# 6. **Education**: Graduates have a higher approval rate than non-graduates
# 7. **Married**: Married applicants show slightly higher approval rates
#
# **Recommendations for modeling:**
# - Use log transforms for income and loan amount
# - Create income-to-loan ratio feature
# - Credit_History should be treated as the most important feature
# - Consider stratified sampling due to mild class imbalance

# %%
print("EDA complete! All plots saved to outputs/ directory.")
