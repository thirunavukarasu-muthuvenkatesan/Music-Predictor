# 🎵 Music Genre Prediction using Decision Tree Classifier

## 📚 Project Description

This project demonstrates a basic implementation of a **Decision Tree Classifier** to predict the **music genre** a person might like based on their **age** and **gender**.

It uses **scikit-learn** for building the model and **pandas** for data handling.

---

## 🛠️ Technologies Used
- Python 3.8+
- pandas
- scikit-learn

---

## 📄 Dataset
- `music-data.csv`
- Features:
  - `age` (numerical)
  - `gender` (0 = female, 1 = male)
- Target:
  - `genre` (music genre label)

---

## 🚀 How It Works

1. **Load Dataset**  
   The music preferences dataset is loaded using pandas.

2. **Prepare Data**  
   - Features (`X`) are selected by dropping the `genre` column.
   - Target (`y`) is the `genre` column.

3. **Split Data**  
   The dataset is split into **training** and **testing** sets (80% training, 20% testing).

4. **Train Model**  
   A **Decision Tree Classifier** is trained using the training data.

5. **Make Predictions**  
   - The model first predicts genres for sample inputs (`[[19,1], [22,0]]`).
   - It then predicts genres for the test set.

6. **Evaluate Model**  
   The model’s performance is evaluated using **accuracy score**.

---

## 🧩 Code Overview
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
music_data = pd.read_csv("music-data.csv")

# Prepare input and output
X = music_data.drop(columns=["genre"])
y = music_data['genre']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
score = accuracy_score(y_test, predictions)
print("Accuracy:", score)
```

---

## 📈 Results

- The model outputs predicted music genres based on age and gender inputs.
- The accuracy score evaluates how well the model performs on unseen data.

---

## 📋 How to Run

1. Make sure you have the required libraries installed:
   ```bash
   pip install pandas scikit-learn
   ```

2. Place your `music-data.csv` file in the project directory.

3. Run the Python script.

---

## 🔥 Future Improvements
- Perform hyperparameter tuning for the Decision Tree.
- Visualize the Decision Tree using `graphviz` or `plot_tree`.
- Try different algorithms like Random Forest, KNN, etc.
- Add more features (like income, education, favorite instruments) to improve prediction accuracy.

---

## 👨‍💻 Author
- **Thirunavukarasu Muthuvenkatesan**
