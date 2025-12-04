import argparse

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

import numpy as np
import pandas as pd

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
income_df.to_csv(output, index=False)