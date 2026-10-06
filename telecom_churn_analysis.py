# Telecom Customer Churn - YuvaIntern Week 1
# Reproducible data exploration and preprocessing workflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# 1. Load data
df = pd.read_csv(DATA_URL)
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all").T)

# 2. Basic data-quality checks
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# 3. Clean TotalCharges
# Some new customers contain blank strings. Convert blanks to NaN, then numeric.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 4. Encode target
df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

# 5. Feature engineering
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

# Count services that are actually subscribed to
df["ServiceCount"] = 0
for c in service_cols:
    df["ServiceCount"] += (df[c] == "Yes").astype(int)

df["TenureBand"] = pd.cut(
    df["tenure"],
    bins=[-1, 6, 12, 36, 72],
    labels=["0-6 months", "7-12 months", "13-36 months", "37+ months"]
)

df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

# 6. Key EDA summaries
overall_churn = df["ChurnFlag"].mean() * 100
contract_churn = df.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
internet_churn = df.groupby("InternetService")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
payment_churn = df.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
tenure_churn = df.groupby("TenureBand", observed=True)["ChurnFlag"].mean().mul(100)

print(f"\nOverall churn: {overall_churn:.2f}%")
print("\nChurn by contract:\n", contract_churn)
print("\nChurn by internet service:\n", internet_churn)
print("\nChurn by payment method:\n", payment_churn)
print("\nChurn by tenure:\n", tenure_churn)

# 7. Visualizations
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.barplot(data=df, x="Contract", y="ChurnFlag", estimator=np.mean)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.barplot(data=df, x="InternetService", y="ChurnFlag", estimator=np.mean)
plt.title("Churn Rate by Internet Service")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag", "ServiceCount"]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 8. Save cleaned data
df.to_csv("telco_customer_churn_cleaned.csv", index=False)
print("\nSaved: telco_customer_churn_cleaned.csv")
