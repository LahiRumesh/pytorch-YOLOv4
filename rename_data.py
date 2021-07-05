import os
import pandas as pd


df = pd.read_csv('data/images/in_data.csv')
image_folder = 'data/images/'
out_csv = 'data/images/out_data.csv'

data_dict={}

for count, filename in enumerate(os.listdir(image_folder)):
    dst ="202" + str(count) + ".jpg"
    data_dict[filename]=dst
    src =image_folder+ filename
    dst =image_folder+ dst

    os.rename(src, dst)

df["image"] = df["image"].map(data_dict)    
df.to_csv(out_csv,index=False)

