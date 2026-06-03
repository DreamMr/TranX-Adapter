from pathlib import Path
import os
from vlmeval.smp import load, dump, md5

DATASET_DIR = os.path.join(Path(__file__).resolve().parent, 'Dataset')

FILE_TRAINING_LIST = [
    r'./Dataset/TranXAdapter-Dataset/training/GenImage_Sdv1d4.jsonl',
    r'./Dataset/TranXAdapter-Dataset/training/GenImageAll.jsonl',
    r'./Dataset/TranXAdapter-Dataset/training/RRDataset.jsonl',
    r'./Dataset/TranXAdapter-Dataset/training/BFP.jsonl'
]

for file_path in FILE_TRAINING_LIST:
    dir_path, file_name = os.path.split(file_path)
    file_name_without_ext, file_ext = os.path.splitext(file_name)
    save_path = os.path.join(dir_path, f"{file_name_without_ext}_processed{file_ext}")
    dataset = load(file_path)
    for data in dataset:
        new_image_path = [os.path.join(DATASET_DIR, p) for p in data['images']]
        data['images'] = new_image_path
        
    dump(dataset, save_path)
    
FILE_EVALUATION_LIST = [
    r'./Dataset/TranXAdapter-Dataset/evaluation/Chamelon.tsv',
    r'./Dataset/TranXAdapter-Dataset/evaluation/GenImage.tsv',
    r'./Dataset/TranXAdapter-Dataset/evaluation/RR.tsv'
]

for file_path in FILE_EVALUATION_LIST:
    dir_path, file_name = os.path.split(file_path)
    file_name_without_ext, file_ext = os.path.splitext(file_name)
    save_path = os.path.join(dir_path, f"{file_name_without_ext}_processed{file_ext}")
    dataset = load(file_path)
    
    dataset['image_path'] = dataset['image_path'].map(lambda p: os.path.join(DATASET_DIR, p))

    dump(dataset, save_path)
    print(f"{save_path} md5: {md5(save_path)}")
