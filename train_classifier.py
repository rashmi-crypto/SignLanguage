import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the data we extracted earlier
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert lists to numpy arrays for the machine learning model
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 2. Split data: 80% for training, 20% for testing the accuracy
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# 3. Initialize and train the "Random Forest" model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 4. Check how well the model learned
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100}% of samples were classified correctly!')

# 5. Save the trained "brain" to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as model.p")