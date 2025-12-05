import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Clean CSV file")
parser.add_argument("--input", type=str, default="../data/adult_income.csv",
                                  help="Input CSV file (default: ../data/adult_income.csv)")
parser.add_argument("--output", type=str, default="../data/adult_income_cleaned.csv",
                                  help="Output CSV file (default: ../data/adult_income_cleaned.csv)")

args = parser.parse_args()

input = args.input
output = args.output

cleaned_rows = []
seen = set()

income_df = pd.read_csv(input)
income_df['workclass'] = income_df['workclass'].astype('string')
income_df['education'] = income_df['education'].astype('string')
income_df['marital.status'] = income_df['marital.status'].astype('string')
income_df['occupation'] = income_df['occupation'].astype('string')
income_df['relationship'] = income_df['relationship'].astype('string')
income_df['race'] = income_df['race'].astype('string')
income_df['sex'] = income_df['sex'].astype('string')
income_df['native.country'] = income_df['native.country'].astype('string')
income_df['income'] = income_df['income'].astype('string')
income_df.replace('?', np.nan, inplace=True)
income_df.dropna(how='any', inplace=True)
income_df.drop_duplicates(inplace=True)

valid_values = {
    'workclass': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
    'education': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'],
    'marital.status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
    'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'],
    'relationship': ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'],
    'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
    'sex': ['Female', 'Male'],
    'native.country': ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'],
    'income': ['<=50K', '>50K']
}

# check for invalid numerical values
numerical_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
for col in numerical_columns:
    if col in income_df.columns:
        try:
            pd.to_numeric(income_df[col], errors='raise')
        except (ValueError, TypeError) as e:
            raise ValueError(f"Column '{col}' contains non-numeric values") from None

# check for invalid cetegorical values
for col, valid_list in valid_values.items():
    if col in income_df.columns:
        invalid_values = income_df[~income_df[col].isin(valid_list)][col].unique()
        if len(invalid_values) > 0:
            raise ValueError(f"Column '{col}' contains invalid value")

income_df.to_csv(output, index=False)
