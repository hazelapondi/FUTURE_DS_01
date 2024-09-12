# Data cleaning & EDA (with Python) of the Titanic Kaggle Dataset

![Titanic Ship](https://github.com/hazelapondi/FUTURE_DS_01/blob/main/img/titanic.jpg)

* Pandas and NumPy are used for data manipulation and mathematical operations, respectively.

* Matplotlib and Seaborn are visualization libraries.

* %matplotlib inline is used to display plots inline in a Jupyter notebook

> From our exploration, we establish among other discoveries that higher-class passengers generally had better survival rates, and older passengers were typically in higher classes.

As part of our data cleaning:

* The Cabin column is dropped because it contains too many missing values.
* Rows with missing Embarked values are also dropped.

When performing Logistic Regression:

* A confusion matrix is created to evaluate the model's predictions.
* The accuracy score is calculated to determine how well the model performed on the test set.

Based on the EDA, these are the key insights gleaned:

* Survival Rate by Gender: Females had a higher survival rate compared to males.
* Survival Rate by Class: Passengers in 1st class had a significantly higher survival rate than those in 3rd class.
* Age Distribution: Most passengers were young adults, with some children and older individuals.
* Embarked Port: Most passengers embarked from Southampton.
