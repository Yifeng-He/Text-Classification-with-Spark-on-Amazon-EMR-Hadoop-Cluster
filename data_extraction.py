
"""
This program is used to extract the review texts from Amazon product dataset, located at
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz
Other product review data sets are available at:
http://jmcauley.ucsd.edu/data/amazon/

1. Download the dataset from the link:
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz
and save it into the local disk

2. Run this Python program to extract the fiels of Overall and ReviewTexts, and then save them into 
a text file: reviews_Automotive.txt
"""

import pandas as pd 
import gzip 

def parse(path): 
    g = gzip.open(path, 'rb') 
    for l in g: yield eval(l) 
    
def getDF(path): 
    i = 0 
    df = {} 
    for d in parse(path): 
        df[i] = d 
        i += 1 
    return pd.DataFrame.from_dict(df, orient='index') 
    
df = getDF('I:\\my_scala_projects\\customer_review_NLP\\reviews_Automotive_5.json.gz')

df_text_ratings = df.loc[:,['overall','reviewText']].values



# save two columns into a text file
file_name = 'reviews_Automotive.txt'
f_out = open(file_name,'w')
for i in range(df_text_ratings.shape[0]):
    line = str(int(df_text_ratings[i,0])) + "\t" + df_text_ratings[i,1] + "\n"
    f_out.write(line)
    line = []
    
f_out.close()

# load the data from the text file
data = pd.read_csv("I:\\my_scala_projects\\customer_review_NLP\\reviews_Automotive.txt", sep='\t', header = None)
# print the first row
print data.iloc[0,0], data.iloc[0,1]

