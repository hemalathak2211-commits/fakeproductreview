# ============================================================
#  train_model.py  —  Train Random Forest with TF-IDF
#  Run once:  python train_model.py
# ============================================================
#
#  What this script does:
#  1. Loads dataset (your CSV or built-in sample data)
#  2. Cleans and preprocesses the review text
#  3. Applies TF-IDF vectorization
#  4. Trains a Random Forest Classifier
#  5. Saves model.pkl and vectorizer.pkl
#  6. Prints accuracy and classification report
#
# ============================================================

import pickle
import re
import numpy  as np
import pandas as pd

from sklearn.ensemble                import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

print("=" * 55)
print("   Fake Review Detection — Random Forest Training")
print("=" * 55)

# ============================================================
#  STEP 1 — Load Dataset
# ============================================================
print("\n📂 STEP 1: Loading Dataset...")

CSV_PATH = None  # Set your CSV path here if you have one
             # Example: CSV_PATH = "fake_reviews.csv"
             # CSV must have columns: "text" and "label"
             # label values: "Fake" or "Genuine"

if CSV_PATH and __import__('os').path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print("✅ Loaded", len(df), "reviews from", CSV_PATH)
else:
    print("ℹ️  No CSV found — using built-in sample dataset")
    print("   Add your own CSV for better accuracy\n")

    genuine_reviews = [
        "The battery life is good and lasts about 2 days on moderate use.",
        "Delivery was fast and packaging was secure. Works as described.",
        "Camera quality is decent for the price. Low light photos are a bit grainy.",
        "Build quality feels solid. Screen is bright and colors look accurate.",
        "Setup was very easy. Connected to wifi in under a minute. No issues so far.",
        "Sound quality is good but bass is slightly lacking for my taste.",
        "Fits perfectly and true to size. Material feels comfortable and durable.",
        "Good value for money. Not perfect but does the job well every day.",
        "Instructions were clear and assembly took about 20 minutes total.",
        "Returned it because color was different from photo. Easy return process.",
        "Using it daily for 3 months with no problems. Very happy with purchase.",
        "Works exactly as described. Fast shipping. Arrived 2 days early.",
        "Quality is acceptable for the price. Nothing exceptional though.",
        "Customer service was responsive when I had a question about it.",
        "Product is slightly smaller than expected but still works fine.",
        "Good product overall. Had a minor issue but it resolved itself quickly.",
        "Solid performance for everyday tasks. Battery drains a bit fast though.",
        "Nice design and very easy to use. Would recommend this to a friend.",
        "Arrived on time. Product matched description. Satisfied with purchase.",
        "Finish looks slightly different in real life but still looks very good.",
        "Works great for the price paid. No complaints after one month of use.",
        "Charging is fast and the display is clear. Happy with this product.",
        "Not the best quality but acceptable. Does what it is supposed to do.",
        "Delivery was delayed by one day but product arrived in perfect condition.",
        "Used it every day for 2 weeks. No issues at all. Good purchase overall.",
    ] * 4

    fake_reviews = [
        "AMAZING AMAZING AMAZING BEST PRODUCT EVER BUY NOW BUY NOW!!!",
        "5 stars 5 stars 5 stars love love love perfect perfect buy buy!!!",
        "WOW WOW WOW THIS IS THE BEST PURCHASE I HAVE EVER MADE IN MY LIFE!!!",
        "Must buy Must buy Must buy Perfect Perfect 100 out of 10 buy now!!!",
        "INCREDIBLE You need this RIGHT NOW DO NOT HESITATE BUY IT NOW!!!",
        "BEST EVER PRODUCT FIVE STARS BUY NOW DONT MISS THIS AMAZING DEAL!!!",
        "Love love love love love buy buy buy buy best best best best!!!",
        "This changed my life Buy it Buy it now You will not regret 5 stars.",
        "AMAZING SUPER WONDERFUL BUY THIS BUY THIS NOW 100 PERCENT RECOMMENDED!!!",
        "Greatest product in history everyone must have this buy now fast shipping!!!",
        "Five stars Five stars Five stars Perfect Perfect Perfect Amazing!!!",
        "BUY NOW DO NOT MISS BEST PRICE BEST QUALITY BEST SERVICE AMAZING DEAL!!!",
        "10 out of 10 The best product ever created Nothing compares Buy it!!!",
        "STRONGLY RECOMMEND BEST IN CATEGORY MUST HAVE EVERYONE NEEDS THIS!!!",
        "Absolutely perfect No issues 100 percent recommend Will buy again!!!",
        "Wonderful wonderful wonderful the best the best amazing amazing!!!",
        "This product is magical Buy it for everyone Life changing product!!!",
        "PERFECT IN EVERY WAY DO NOT THINK JUST BUY ZERO REGRETS LOVE IT!!!",
        "Best product best quality best delivery best service best best buy!!!",
        "HIGHLY HIGHLY RECOMMENDED AMAZING VALUE BUY NOW FIVE STARS!!!",
        "Perfect perfect perfect five stars five stars love love buy buy now!!!",
        "BEST PURCHASE EVER EVER EVER DO NOT MISS THIS INCREDIBLE DEAL NOW!!!",
        "Amazing amazing amazing product love love love must must must buy!!!",
        "WOW incredible product best best best buy buy buy love love love!!!",
        "DONT WAIT BUY NOW BUY NOW BUY NOW BEST PRODUCT AMAZING AMAZING!!!",
    ] * 4

    data = (
        [(r, "Genuine") for r in genuine_reviews] +
        [(r, "Fake")    for r in fake_reviews]
    )
    np.random.shuffle(data)
    df = pd.DataFrame(data, columns=["text", "label"])

print("✅ Dataset loaded:", len(df), "reviews")
print("   Classes:", df["label"].value_counts().to_dict())

# ============================================================
#  STEP 2 — Preprocess Text
# ============================================================
print("\n🧹 STEP 2: Preprocessing Text...")

def preprocess(text):
    text = str(text).lower()                        # lowercase
    text = re.sub(r'[^a-z0-9\s]', ' ', text)       # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()        # remove extra spaces
    return text

df["clean_text"] = df["text"].apply(preprocess)
print("✅ Text preprocessing complete")
print("   Example original:", df["text"].iloc[0][:50])
print("   Example cleaned: ", df["clean_text"].iloc[0][:50])

# ============================================================
#  STEP 3 — TF-IDF Vectorization
# ============================================================
print("\n📊 STEP 3: Applying TF-IDF Vectorization...")
print("   TF-IDF = Term Frequency - Inverse Document Frequency")
print("   Converts text into numerical feature vectors")

vectorizer = TfidfVectorizer(
    max_features = 5000,       # use top 5000 most important words
    ngram_range  = (1, 2),     # use single words AND word pairs
    stop_words   = 'english',  # remove common words like 'the', 'is'
    min_df       = 2           # word must appear at least 2 times
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("✅ TF-IDF vectorization complete")
print("   Total features (words):", X.shape[1])
print("   Total samples:", X.shape[0])

# ============================================================
#  STEP 4 — Train / Test Split
# ============================================================
print("\n✂️  STEP 4: Splitting Dataset into Train and Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,     # 20% for testing
    random_state = 42,      # fixed seed for same results
    stratify     = y        # equal distribution of classes
)

print("✅ Dataset split complete")
print("   Training samples:", X_train.shape[0])
print("   Testing samples: ", X_test.shape[0])

# ============================================================
#  STEP 5 — Train Random Forest Model
# ============================================================
print("\n🌲 STEP 5: Training Random Forest Classifier...")
print("   Random Forest = 100 Decision Trees working together")
print("   Each tree votes → majority vote = final prediction")

model = RandomForestClassifier(
    n_estimators = 100,    # number of decision trees
    max_depth    = None,   # trees can grow as deep as needed
    random_state = 42,     # fixed seed for same results
    n_jobs       = -1      # use all CPU cores for speed
)

model.fit(X_train, y_train)
print("✅ Random Forest training complete")
print("   Total trees trained:", model.n_estimators)

# ============================================================
#  STEP 6 — Evaluate Model
# ============================================================
print("\n📈 STEP 6: Evaluating Model Performance...")

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print("✅ Evaluation complete")
print("\n" + "─" * 45)
print("   MODEL ACCURACY:", round(acc * 100, 2), "%")
print("─" * 45)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("              Predicted")
print("              Fake  Genuine")
print("Actual Fake  ", cm[0])
print("Actual Genuine", cm[1])

# Feature importance
print("\nTop 10 Important Words:")
feature_names = vectorizer.get_feature_names_out()
importances   = model.feature_importances_
top_indices   = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(top_indices):
    print("  ", i+1, "→", feature_names[idx],
          "( importance:", round(importances[idx], 4), ")")

# ============================================================
#  STEP 7 — Save Model Files
# ============================================================
print("\n💾 STEP 7: Saving Model Files...")

with open("model.pkl",      "wb") as f: pickle.dump(model,      f)
with open("vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)

print("✅ Saved: model.pkl")
print("✅ Saved: vectorizer.pkl")

print("\n" + "=" * 55)
print("   Training Complete!")
print("   Accuracy:", round(acc * 100, 2), "%")
print("   Now run:  python app.py")
print("=" * 55)