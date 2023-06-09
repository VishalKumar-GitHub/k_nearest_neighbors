# k_nearest_neighbors

This project is a Python implementation of a K-nearest neighbors classifier using the MNIST dataset.
# Index

- Built With
- Prerequisites
- Getting Started
- Usage
- Algorithm
- Data
- Feedback
- Contribution
- Author
- License
- 
## Built With
- Python
- Pandas
- Scikit-learn

## Prerequisites
To run this project, you need to have the following installed:
- Python 
- Pandas 
- Scikit-learn 

## Getting Started
To get started with this project, follow the instructions below:

1. Clone the repository:
   ```
   git clone https://github.com/VishalKumar-GitHub/k_nearest_neighbors.git
   ```

2. Navigate to the project directory:
   ```
   cd k_nearest_neighbors
   ```

3. Install the required dependencies:
   ```
   pip install pandas scikit-learn
   ```

4. Run the Python script:
   ```
   python your_script.py
   ```

## Usage
To use this project, follow these steps:

1. Make sure you have the MNIST dataset in a CSV file named `mnist.csv`.

2. Import the required libraries:
   ```python
   import pandas as pd
   import sklearn.model_selection
   import sklearn.neighbors
   from sklearn.neighbors import KNeighborsClassifier
   import sklearn.metrics
   ```

3. Load the dataset:
   ```python
   df = pd.read_csv("mnist.csv")
   df = df.set_index('id')
   ```

4. Split the dataset into training and testing sets:
   ```python
   x = df.drop(["class"], axis=1)
   y = df["class"] 
   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)
   ```

5. Create and train the K-nearest neighbors classifier:
   ```python
   neigh = KNeighborsClassifier(n_neighbors=5)
   neigh.fit(x_train, y_train)
   ```

6. Make predictions on the test set and calculate accuracy:
   ```python
   y_predicted = neigh.predict(x_test)
   accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
   ```

7. Use the predicted values for your desired purposes.

## Features
- Load MNIST dataset from a CSV file
- Split the dataset into training and testing sets
- Train a K-nearest neighbors classifier
- Predict the classes of test instances
- Calculate accuracy of the classifier

## Algorithm
This project implements the K-nearest neighbors algorithm. It works by finding the K nearest neighbors of a test instance in the training set and assigning the class label that appears most frequently among the neighbors.

## Data
The project uses the MNIST dataset, which is a widely used benchmark dataset in the field of machine learning. It consists of grayscale images of handwritten digits (0-9) along with their corresponding labels.

## Feedback
If you have any feedback, suggestions, or issues regarding this project, please [open an issue](https://github.com/VishalKumar-GitHub/k_nearest_neighbors/issues).


## Contribution
Contributions to this project are welcome. If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request describing your changes.

## Author
- Vishal Kumar
- GitHub: [VishalKumar-GitHub](https://github.com/VishalKumar-GitHub)

## License
This project is licensed under the MIT License. See the [LICENSE.md](https://github.com/VishalKumar-GitHub/k_nearest_neighbors/blob/main/LICENSE) file for details.
