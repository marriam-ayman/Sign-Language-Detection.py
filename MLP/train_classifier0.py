import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
from sklearn.neural_network import MLPClassifier

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Pad sequences to a fixed length (assuming max_length is known)
max_length = max(len(seq) for seq in data)
data_padded = np.zeros((len(data), max_length))
for i, seq in enumerate(data):
    data_padded[i, :len(seq)] = seq

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = MLPClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy:', accuracy)

# Calculate loss
loss = 1 - accuracy
print('Loss:', loss)

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, y_predict)
print('Balanced Accuracy:', balanced_acc)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
