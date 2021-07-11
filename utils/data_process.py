from pycocotools.coco import COCO
import csv
import os
import pandas as pd
import glob
import shutil


def GetFileList(dirName,endings=['.jpg','.jpeg','.png','.mp4']):
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for i,ending in enumerate(endings):
        if ending[0]!='.':
            endings[i] = '.'+ending
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetFileList(fullPath,endings)
        else:
            for ending in endings:
                if entry.endswith(ending):
                    allFiles.append(fullPath)               
    return allFiles  



def coco_data_convert(coco_json,coco_folder,skip_background=True,check_file=True):


    coco = COCO(coco_json)

    if check_file:
        imgIds = coco.getImgIds()
        for imgId in imgIds:
            img = coco.loadImgs(imgId)[0]
            if not os.path.isfile(os.path.join(coco_folder, img['file_name'])):
                print('The image {} is not exist.'.format(img['file_name']))
                exit()

    data_field=[]
    annIds = coco.getAnnIds()
    for annId in annIds:
        ann = coco.loadAnns(annId)[0]
        img = coco.loadImgs(int(ann['image_id']))[0]
        cat = coco.loadCats(ann['category_id'])[0]

        if cat['name'] == 'background' and skip_background == True:
            continue

        x1,y1,w,h = ann['bbox']
        x2 = x1 + w
        y2 = y1 + h

        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        data_field.append([img['file_name'], str(x1), str(y1), str(x2), str(y2), cat['name']])

    cols = ["image","xmin","ymin","xmax","ymax","label"]
    data_frame = pd.DataFrame(data_field, columns=cols)

    return data_frame


def create_unique_id(df,image_folder,index_no='400',out_folder="train_set"):

    data_dict={}
    dir_path = os.path.dirname(image_folder)
    output_path = os.path.join(dir_path, out_folder)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_paths = GetFileList(image_folder)

    for img in input_paths:
        shutil.copy(img, output_path)


    for count, filename in enumerate(os.listdir(output_path)):
        dst = index_no + str(count) + ".jpg"
        data_dict[filename] = dst
        src = os.path.join(output_path, filename)
        dst = os.path.join(output_path, dst)

        
        os.rename(src, dst)

    df["image"] = df["image"].map(data_dict)
    
    return df,output_path




def convert_Input_CSV_to_yolo(vott_df,labeldict=dict(zip(['Object'],[0,])),path='',target_name='train.txt',abs_path=False):
    if not 'code' in vott_df.columns:
        vott_df['code']=vott_df['label'].apply(lambda x: labeldict[x])
    for col in vott_df[['xmin', 'ymin', 'xmax', 'ymax']]:
        vott_df[col]=(vott_df[col]).apply(lambda x: round(float(x)))
        
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



def pre_process(folder_path,folder_name='train',class_names='data/class.names',
                json_file='via_export_coco.json',train_file ='data/train.txt',val_file = 'data/val.txt',val_split=0.9):


    image_folder_path=os.path.join(folder_path,folder_name)
    json_file_path = os.path.join(image_folder_path,json_file )
    
    df = coco_data_convert(json_file_path, image_folder_path)
    multi_df, out_data_dir = create_unique_id(df, image_folder_path)

    labels = multi_df['label'].unique()
    labeldict = dict(zip(labels,range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep='first', inplace=True)
    
    convert_Input_CSV_to_yolo(multi_df,labeldict,path = out_data_dir,target_name=train_file)
    file = open(class_names,"w") 
    SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])
    for elem in SortedLabelDict:
        file.write(elem[0]+'\n') 
    file.close() 

    dataset = pd.read_csv(train_file,delimiter="\n")
    dataset_copy = dataset.copy()
    val_set = dataset_copy.sample(frac=val_split, random_state=0)
    val_set.to_csv(val_file, header=True, index=False, sep='\n', mode='w')


    return out_data_dir


#data_prepare(folder_path)
#data_prepare(folder_path, 'test_set','data/test.names','test','via_export_coco.json','data/val.txt')    

