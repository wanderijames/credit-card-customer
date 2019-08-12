# Udacity Nanodegree, Machine learning

### Capstone project

The data used can be downloaded from [Kaggle](https://www.kaggle.com/arjunbhasin2013/ccdata)

Place that in the root directory as `ccdata.zip`

#### Development Environment Setup
1. Install `pipenv` as documented in their [official page](https://docs.pipenv.org/en/latest/)
2. While in the root directory install the python virtual environment:

	> pipenv --python 3.6

3. Install the dependencies:

	> pipenv install --dev

	If you want to use the pip for your setup, you generate the requirements.txt file:

	> pipenv lock -r > requirements.txt

4. Start the Jupyter notebook

	> jupyter notebook credit-card-customer.ipynb