from ydata_profiling import ProfileReport
import pandas as pd
from itertools import chain
from collections import Counter
import hashlib

from utils import get_table

# md5hash and md5hash with prefix are needed to create unique global id for each node type 
def md5hash(s: str): 
    return hashlib.md5(str(s).encode('utf-8')).hexdigest()

def getmd5hash(prefix):
    return lambda x: md5hash(prefix + x)
    


business_df = get_table('yelp_dataset/yelp_academic_dataset_business.json')

# profile report is useful to get first insights on the dataset
profile = ProfileReport(business_df, title="Businesses Report")
profile.to_file("businesses_report.html")

print(business_df[business_df['categories'].isna()])
profile = ProfileReport(business_df[business_df['categories'].isna()], title="Businesses Report Category na")
profile.to_file("businesses_report_category_na.html")

business_df = business_df[business_df['categories'].notna()]
business_df['business_id'] = business_df['business_id'].apply(getmd5hash('business'))
business_df.insert(loc=0, column=':LABEL', value=['Business']*len(business_df))

# # CATEGORY
categories = business_df['categories'].to_list()
categories = set(chain.from_iterable([[cat.strip() for cat in c.split(',')] for c in categories]))
categories_df = pd.DataFrame(categories, columns=['name'])
categories_df.insert(loc=0, column='category_id:ID', value=categories_df['name'].apply(md5hash))
categories_df.insert(loc=2, column=':LABEL', value=['Category']*len(categories_df))

business_df.rename(columns={"business_id": "business_id:ID"}, errors='raise', inplace=True)

category_business_rels = []
for _, row in business_df.iterrows():
    for category in row['categories'].split(','):
        category = category.strip()
        category_id = md5hash(category)
        business_id = row['business_id:ID']
        category_business_rels.append([business_id, category_id, 'HAS_CATEGORY'])

category_business_rels_df = pd.DataFrame(category_business_rels, columns=[':START_ID', ':END_ID', ':TYPE'])

# # USER
user_df = get_table('yelp_dataset/yelp_academic_dataset_user.json')
user_df.rename(columns={"user_id": "user_id:ID"}, errors='raise', inplace=True)
user_df.insert(loc=22, column=':LABEL', value=['User']*len(user_df))

# # FRIENDS
friends = user_df[['user_id:ID', 'friends']].copy()
friends['friends'] = friends['friends'].str.split(',')
friends = friends.explode('friends')
friends = friends[friends['friends'].isin(friends['user_id:ID'])]
friends = pd.concat([friends, friends.rename(columns={"friends": "user_id:ID", "user_id:ID": "friends"}, errors="raise")]).drop_duplicates(subset = ["friends", "user_id:ID"])
friends.insert(loc=2, column=':TYPE', value=['FRIEND_OF']*len(friends))
friends.rename(columns={"user_id:ID": ":START_ID", "friends":":END_ID"}, errors='raise', inplace=True)


# # REVIEW
review_df = get_table('yelp_dataset/yelp_academic_dataset_review.json')
review_df['business_id'] = review_df['business_id'].apply(getmd5hash('business'))
review_df = review_df[review_df['business_id'].isin(business_df['business_id:ID'])]
review_df = review_df[review_df['user_id'].isin(user_df['user_id:ID'])]

review_df.insert(loc=0, column=':LABEL', value=['Review']*len(review_df))

review_df['text'] = review_df['text'].str.replace('[",\\n,\\r]', '')
review_df['review_id'] = review_df['review_id'].apply(getmd5hash('review')) # we need global ids (review_id and user_id share some values)


review_business_rels_df = review_df[['review_id', 'business_id']].copy().rename(columns={"review_id": ":START_ID", "business_id": ":END_ID"}, errors='raise')
review_business_rels_df.insert(loc=2, column=':TYPE', value=['REVIEW_BUSINESS']*len(review_df))

user_review_rels_df = review_df[['user_id', 'review_id']].copy().rename(columns={"user_id": ":START_ID", "review_id": ":END_ID"}, errors='raise')
user_review_rels_df.insert(loc=2, column=':TYPE', value=['USER_REVIEW']*len(review_df))

review_df.rename(columns={"review_id": "review_id:ID",'stars':'stars:float','useful':'useful:float','funny':'funny:float','cool':'cool:float'}, errors='raise', inplace=True)
review_df.drop(columns=['user_id', 'business_id'], inplace=True)

# # TIP
tip_df = get_table('yelp_dataset/yelp_academic_dataset_tip.json')
tip_df['business_id'] = tip_df['business_id'].apply(getmd5hash('business'))
tip_df = tip_df[tip_df['business_id'].isin(business_df['business_id:ID'])]
tip_df = tip_df[tip_df['user_id'].isin(user_df['user_id:ID'])]
tip_df.insert(loc=0, column=':LABEL', value=['Tip']*len(tip_df))

tip_df['index'] = tip_df.index
tip_df.insert(loc=0, column='tip_id', value=tip_df['index'].apply(md5hash))


tip_df['text'] = tip_df['text'].str.replace('[",\\n,\\r]', '')
tip_business_rels_df = tip_df[['tip_id', 'business_id']].copy().rename(columns={"tip_id": ":START_ID", "business_id": ":END_ID"}, errors='raise')
tip_business_rels_df.insert(loc=2, column=':TYPE', value=['TIP_BUSINESS']*len(tip_df))

user_tip_rels_df = tip_df[['user_id', 'tip_id']].copy().rename(columns={"user_id": ":START_ID", "tip_id": ":END_ID"}, errors='raise')
user_tip_rels_df.insert(loc=2, column=':TYPE', value=['USER_TIP']*len(tip_df))

tip_df.rename(columns={"tip_id": "tip_id:ID"}, errors='raise', inplace=True)
tip_df.drop(columns=['index', 'user_id', 'business_id'], inplace=True)


business_df.to_csv('neo4j_csvs/business.csv', index=False)
categories_df.to_csv('neo4j_csvs/categories.csv', index=False)
category_business_rels_df.to_csv('neo4j_csvs/category_business_rels.csv', index=False)
user_df.to_csv('neo4j_csvs/user.csv', index=False)

friends.to_csv('neo4j_csvs/user_user_rels.csv', index=False)

review_df.to_csv('neo4j_csvs/review.csv', index=False)
review_business_rels_df.to_csv('neo4j_csvs/review_business_rels.csv', index=False)
user_review_rels_df.to_csv('neo4j_csvs/user_review_rels.csv', index=False)

tip_df.to_csv('neo4j_csvs/tip.csv', index=False)
tip_business_rels_df.to_csv('neo4j_csvs/tip_business_rels.csv', index=False)
user_tip_rels_df.to_csv('neo4j_csvs/user_tip_rels.csv', index=False)

# COMMAND
# bin/neo4j-admin database import full --nodes=import/business.csv --nodes=import/categories.csv --nodes=import/review.csv --nodes=import/tip.csv --nodes=import/user.csv --relationships=import/category_business_rels.csv --relationships=import/review_business_rels.csv --relationships=import/tip_business_rels.csv --relationships=import/user_review_rels.csv --relationships=import/user_tip_rels.csv --relationships=import/user_user_rels.csv --overwrite-destination neo4j