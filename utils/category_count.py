from itertools import chain
from collections import Counter
import pandas as pd

from utils import get_table

business_df = get_table('yelp_dataset/yelp_academic_dataset_business.json')


categories = business_df['categories'].dropna().to_list()

categories = [[cat.strip() for cat in c.split(',')] for c in categories]

c = Counter(chain(*categories))

df = pd.Series(c).sort_index().rename_axis('nome').reset_index(name='count')
df = df[df['count'] > 100]

print(df.sort_values(by='count', ascending=False))