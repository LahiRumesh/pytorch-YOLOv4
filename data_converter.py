from os import path, makedirs
import os
import pandas as pd
import sys
import argparse




IMAGE_FOLDER = 'data/images'
DATA_CSV_FILE =  'images/out_data.csv'
YOLO_FILE = 'data/train.txt'
DATA_CLASSES='data/class.names'


def convert_Input_CSV_to_yolo(vott_df,labeldict=dict(zip(['Object'],[0,])),path='',target_name='data_train.txt',abs_path=False):

    if not 'code' in vott_df.columns:
        vott_df['code']=vott_df['label'].apply(lambda x: labeldict[x])
    for col in vott_df[['xmin', 'ymin', 'xmax', 'ymax']]:
        vott_df[col]=(vott_df[col]).apply(lambda x: round(x))
        
    #Create Yolo Text file
    last_image = ''
    txt_file = ''

    for index,row in vott_df.iterrows():
        if not last_image == row['image']:
            if abs_path:
                txt_file +='\n'+row['image_path'] + ' '
            else:
                txt_file +='\n'+os.path.join(path,row['image']) + ' '
            txt_file += ','.join([str(x) for x in (row[['xmin', 'ymin', 'xmax', 'ymax','code']].tolist())])
        else:
            txt_file += ' '
            txt_file += ','.join([str(x) for x in (row[['xmin', 'ymin', 'xmax', 'ymax','code']].tolist())])
        last_image = row['image']
    file = open(target_name,"w") 
    file.write(txt_file[1:]) 
    file.close() 
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--IMAGE_FOLDER", type=str, default=IMAGE_FOLDER,help = "Image folder path")
    parser.add_argument("--DATA_CSV_FILE", type=str, default=DATA_CSV_FILE,help = "Input CSV file that store data")
    parser.add_argument("--YOLO_FILE", type=str, default=YOLO_FILE,help = "data train yolo file name"  )

    args = parser.parse_args()
    multi_df = pd.read_csv(args.DATA_CSV_FILE)
    labels = multi_df['label'].unique()
    labeldict = dict(zip(labels,range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep='first', inplace=True)
    train_path = args.IMAGE_FOLDER
    convert_Input_CSV_to_yolo(multi_df,labeldict,path = train_path,target_name=args.YOLO_FILE)

    file = open(DATA_CLASSES,"w") 
    SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])
    for elem in SortedLabelDict:
	    file.write(elem[0]+'\n') 
    file.close() 

