# ============================================================
#  TASK 1: Iris Flower Classification
#  CodeAlpha Data Science Internship
#  Author: Muhammad Zaeem
# ============================================================
#
#  OBJECTIVE:
#  Train and evaluate a machine learning model to classify
#  Iris flower species (Setosa, Versicolor, Virginica) based
#  on sepal and petal measurements.
#
#  DATASET FEATURES:
#  - SepalLengthCm, SepalWidthCm
#  - PetalLengthCm, PetalWidthCm
#  - Species (target label)
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── 2. LOAD DATASET ──────────────────────────────────────────
print("=" * 55)
print("  IRIS FLOWER CLASSIFICATION — CodeAlpha Task 1")
print("=" * 55)

df = pd.read_csv("Iris.csv")

print(f"\n📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n🔍 First 5 rows:")
print(df.head())

# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n🌸 Species Distribution:")
print(df["Species"].value_counts())

print("\n✅ Missing Values:")
print(df.isnull().sum())

# ── 4. VISUALIZATIONS ────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
colors = ["#4CAF50", "#2196F3", "#FF5722"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
for ax, feature in zip(axes.flatten(), features):
    for i, species in enumerate(df["Species"].unique()):
        subset = df[df["Species"] == species][feature]
        ax.hist(subset, bins=15, alpha=0.7, label=species, color=colors[i])
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("iris_eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📸 Saved: iris_eda_distributions.png")

# Pairplot — relationships between all features
fig2 = sns.pairplot(
    df.drop(columns=["Id"]),
    hue="Species",
    palette={"Iris-setosa": "#4CAF50", "Iris-versicolor": "#2196F3", "Iris-virginica": "#FF5722"},
    diag_kind="kde",
    plot_kws={"alpha": 0.7}
)
fig2.fig.suptitle("Iris — Feature Pair Relationships", y=1.02, fontsize=14, fontweight="bold")
fig2.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("📸 Saved: iris_pairplot.png")

# Correlation heatmap
fig3, ax3 = plt.subplots(figsize=(8, 6))
corr = df.drop(columns=["Id", "Species"]).corr()
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax3, vmin=-1, vmax=1,
    annot_kws={"size": 12}
)
ax3.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("iris_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("📸 Saved: iris_correlation_heatmap.png")

# Boxplot for each feature by species
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle("Feature Distribution by Species (Boxplots)", fontsize=15, fontweight="bold")
for ax, feature in zip(axes4.flatten(), features):
    sns.boxplot(
        data=df, x="Species", y=feature, ax=ax,
        palette={"Iris-setosa": "#4CAF50", "Iris-versicolor": "#2196F3", "Iris-virginica": "#FF5722"}
    )
    ax.set_title(feature)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("iris_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("📸 Saved: iris_boxplots.png")

# ── 5. PREPROCESSING ─────────────────────────────────────────
# Drop ID column (not a feature)
df = df.drop(columns=["Id"])

# Encode species labels to numbers
le = LabelEncoder()
df["Species_Encoded"] = le.fit_transform(df["Species"])
print(f"\n🔢 Label Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Define features (X) and target (y)
X = df[features]
y = df["Species_Encoded"]

# Scale features for distance-based models (KNN, SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📂 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 6. MODEL TRAINING & COMPARISON ───────────────────────────
print("\n🤖 Training 4 models...")

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel="rbf", C=1.0, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    results[name] = {
        "model":    model,
        "accuracy": acc,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "y_pred":   y_pred,
    }
    print(f"  ✅ {name}: Test Acc={acc*100:.2f}%  |  CV={cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── 7. BEST MODEL — DETAILED EVALUATION ─────────────────────
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name}")
print(f"   Test Accuracy  : {best['accuracy']*100:.2f}%")
print(f"   CV Accuracy    : {best['cv_mean']*100:.2f}% ± {best['cv_std']*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, best["y_pred"], target_names=le.classes_))

# ── 8. VISUALIZATIONS — MODEL RESULTS ────────────────────────
# Model comparison bar chart
fig5, ax5 = plt.subplots(figsize=(10, 5))
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] * 100 for m in model_names]
cv_means    = [results[m]["cv_mean"]  * 100 for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = ax5.bar(x - width/2, accuracies, width, label="Test Accuracy", color="#2196F3", alpha=0.85)
bars2 = ax5.bar(x + width/2, cv_means,   width, label="CV Accuracy",   color="#4CAF50", alpha=0.85)

ax5.set_xticks(x)
ax5.set_xticklabels([m.replace(" (SVM)", "\n(SVM)") for m in model_names], fontsize=10)
ax5.set_ylabel("Accuracy (%)", fontsize=12)
ax5.set_title("Model Comparison — Test vs Cross-Validation Accuracy", fontsize=13, fontweight="bold")
ax5.set_ylim(85, 102)
ax5.legend()
ax5.yaxis.grid(True, linestyle="--", alpha=0.7)

for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("📸 Saved: iris_model_comparison.png")

# Confusion matrix for best model
cm = confusion_matrix(y_test, best["y_pred"])
fig6, ax6 = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax6, cmap="Blues", colorbar=False)
ax6.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("📸 Saved: iris_confusion_matrix.png")

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
fig7, ax7 = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(importances)[::-1]
bars = ax7.bar(
    [features[i] for i in sorted_idx], importances[sorted_idx],
    color=["#FF5722", "#FF5722", "#2196F3", "#2196F3"],
    alpha=0.85, edgecolor="white"
)
ax7.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
ax7.set_ylabel("Importance Score")
ax7.yaxis.grid(True, linestyle="--", alpha=0.6)
for bar in bars:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("📸 Saved: iris_feature_importance.png")

# ── 9. PREDICT A NEW SAMPLE ──────────────────────────────────
print("\n🌺 Predicting a new flower sample...")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # Likely Setosa
sample_scaled = scaler.transform(sample)
prediction = best["model"].predict(sample_scaled)
predicted_species = le.inverse_transform(prediction)[0]
print(f"   Input  : SepalL=5.1, SepalW=3.5, PetalL=1.4, PetalW=0.2")
print(f"   Predicted Species: ✅ {predicted_species}")

print("\n" + "=" * 55)
print("  ✅ TASK 1 COMPLETE — All outputs saved!")
print("=" * 55)
