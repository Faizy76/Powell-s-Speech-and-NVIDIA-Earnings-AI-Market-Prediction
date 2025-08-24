

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- # Step 1: Create Powell Speech Dataset 
data = {
    "date": [
        "2022-06-15","2022-07-27","2022-09-21","2023-03-22","2023-07-26","2023-12-13","2024-06-12","2025-08-22"
    ],
    "speech_text": [
        "High inflation risk, aggressive hikes needed",
        "Rate hikes slowing, economy resilient",
        "Inflation elevated, policy restrictive",
        "Bank stress monitored, rate hikes near end",
        "Soft landing possible, rate cuts likely",
        "Inflation falling, cuts possible in 2024",
        "Rates high but cuts under consideration",
        "Economy slowing, rate cuts possible, markets respond positively"
    ],
    "sentiment_score": [-0.7, 0.6, -0.8, 0.4, 0.9, 0.5, 0.3, 0.8],
    "sp500_change": [-2.1, 1.8, -1.5, 1.3, 2.5, 1.0, 0.7, 0.9],
    "btc_change": [-3.5, 2.2, -1.9, 1.9, 3.1, 1.2, 0.5, 0.7],
    "label": [0, 1, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv("powell_speeches.csv", index=False)
print(" Powell speeches dataset saved as powell_speeches.csv")

# --- Step 2: Features (speech_text only for NLP)
X_text = df["speech_text"]
y = df["label"]

# --- Step 3: Build pipeline (TF-IDF + Logistic Regression)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=200)),
    ("clf", LogisticRegression(max_iter=200))
])

# --- Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42)

# --- Step 5: Train model
pipe.fit(X_train, y_train)

# --- Step 6: Evaluate
preds = pipe.predict(X_test)
print("\n📊 Model Performance:\n", classification_report(y_test, preds))

# --- Step 7: Live prediction example
new_speech = ["Rate cuts likely, inflation cooling"]
pred = pipe.predict(new_speech)[0]
print("\n New Powell speech prediction:", "Market UP " if pred==1 else "Market DOWN ")

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Prepare the dataset
data = {
    "date": [
        "2022-06-15","2022-07-27","2022-09-21","2023-03-22",
        "2023-07-26","2023-12-13","2024-06-12","2025-08-22"
    ],
    "speech_text": [
        "High inflation risk, aggressive hikes needed",
        "Rate hikes slowing, economy resilient",
        "Inflation elevated, policy restrictive",
        "Bank stress monitored, rate hikes near end",
        "Soft landing possible, rate cuts likely",
        "Inflation falling, cuts possible in 2024",
        "Rates high but cuts under consideration",
        "Economy slowing, rate cuts possible, markets respond positively"
    ],
    "sentiment_score": [-0.7, 0.6, -0.8, 0.4, 0.9, 0.5, 0.3, 0.8],
    "sp500_change": [-2.1, 1.8, -1.5, 1.3, 2.5, 1.0, 0.7, 0.9],
    "btc_change": [-3.5, 2.2, -1.9, 1.9, 3.1, 1.2, 0.5, 0.7],
    "label": [0, 1, 0, 1, 1, 1, 1, 1]  # 1 = rate cut likely (good), 0 = no cut (bad)
}

df = pd.DataFrame(data)

# Step 3: Features and labels
X = df[["sentiment_score","sp500_change","btc_change"]]
y = df["label"]

# Optional: scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict probability for a new upcoming September speech
# Example: Assume sentiment_score=0.85, SP500 change=1.2%, BTC change=0.6%
new_speech = [[0.85, 1.2, 0.6]]
new_speech_scaled = scaler.transform(new_speech)
prob_rate_cut = model.predict_proba(new_speech_scaled)[0][1]  # probability of rate cut (good)
prob_no_cut = 1 - prob_rate_cut  # probability of no cut (bad)

print(f"Predicted probability of planned cut in rates (GOOD Result): {prob_rate_cut*100:.2f}%")
print(f"Predicted probability of No planned decrease in rates (BAD Result): {prob_no_cut*100:.2f}%")

# Step 7: Visualize probabilities with a bar plot
plt.figure(figsize=(6,4))
sns.barplot(x=["Rate Cut (Good)","No Cut (Bad)"],
            y=[prob_rate_cut*100, prob_no_cut*100],
            palette="viridis")
plt.ylim(0,100)
plt.ylabel("Probability (%)")
plt.title("Powell September Speech Prediction")
plt.show()

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Sample historical NVIDIA quarterly data + Crypto reaction
data = {
    "quarter": ["2024Q1","2024Q2","2024Q3","2024Q4","2025Q1","2025Q2"],
    "sentiment_score": [0.5, 0.7, -0.3, 0.2, 0.6, 0.4],
    "sp500_change": [1.0, 1.2, -0.5, 0.3, 0.8, 0.5],
    "nvidia_change": [2.1, 3.0, -1.2, 0.5, 2.5, 1.0],
    "btc_change": [1.5, 2.0, -1.0, 0.3, 1.8, 0.7],  # BTC % change after earnings
    "label": [1, 1, 0, 1, 1, 1]  # 1=UP, 0=DOWN for Nvidia
}

df = pd.DataFrame(data)

# Step 3: Features and labels for Nvidia prediction
X = df[["sentiment_score","sp500_change"]]
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict probability for upcoming Wednesday result
new_quarter = [[0.65, 0.9]]  # sentiment_score=0.65, SP500 change=0.9%
new_quarter_scaled = scaler.transform(new_quarter)
prob_up = model.predict_proba(new_quarter_scaled)[0][1]
prob_down = 1 - prob_up

# Step 7: Estimate Crypto impact based on Nvidia prediction
# Simple assumption: crypto follows Nvidia movement with slight dampening
crypto_prob_up = prob_up * 0.7  # 70% correlation
crypto_prob_down = 1 - crypto_prob_up

print(f"Nvidia Probability Up: {prob_up*100:.2f}%")
print(f"Nvidia Probability Down: {prob_down*100:.2f}%")
print(f"Crypto Market Probability Up: {crypto_prob_up*100:.2f}%")
print(f"Crypto Market Probability Down: {crypto_prob_down*100:.2f}%")
print(f"BTC Probability Up: {crypto_prob_up*100:.2f}%")
print(f"BTC Market Probability Down: {crypto_prob_down*100:.2f}%")

# Step 8: Visualize both Nvidia and Crypto probabilities
plt.figure(figsize=(8,5))
sns.barplot(
    x=["Nvidia Up","Nvidia Down","Crypto Up","Crypto Down"],
    y=[prob_up*100, prob_down*100, crypto_prob_up*100, crypto_prob_down*100],
    palette="coolwarm"
)
plt.ylim(0,100)
plt.ylabel("Probability (%)")
plt.title("Impact of Nvidia Earnings on Stock & Crypto")
plt.show()

