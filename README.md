# Data cleaning & EDA (with Python) of the Kaggle Dataset

* Pandas and NumPy are used for data manipulation and mathematical operations, respectively.

* Matplotlib and Seaborn are visualization libraries.

* %matplotlib inline is used to display plots inline in a Jupyter notebook

> From our exploration, we establish among aother discoveries, that higher-class passengers generally had better survival rates and typically, older passengers were in higher classes.

As part of our data cleaning:

* The Cabin column is dropped because it contains too many missing values.
* Rows with missing Embarked values are also dropped.

When performing Logistic Regression:

* A confusion matrix is created to evaluate the model's predictions.
* The accuracy score is calculated to determine how well the model performed on the test set.
