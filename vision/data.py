import os 
import re 
from random import randint 

from tqdm import tqdm 
from random import shuffle 
from itertools import count, copyfile 
from shutil import rmtree 
import pandas as pd 
import json 

import h5py 
import torch 

import numpy as np 
import cv2 

from configs import *


class _RepeatSampler(object):

    # https://github.com/pytorch/pytorch/issues/15849

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    # https://github.com/pytorch/pytorch/issues/15849

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class FastH5:
    def __init__(self, h5file, total_data):
        self.file = h5file
        self.group_size = int(total_data**0.5)
        self.counter = 0
        self.group_number = 0
        self.current_group = h5file.create_group(f"group_{self.group_number}")

    def create_dataset(self, name, data, **kwargs):
        # h5files suffer significantly when all datasets are in root folder
        # same as a filesystem will suffer when theres 1M files in the same folder
        # this allows O(n^2) reduction in # of dataset in each group
        # it is possible to make it O(log(n)), but like, why

        if self.counter > self.group_size:
            self.counter = 0
            self.group_number += 1
            self.current_group = self.file.create_group(f"group_{self.group_number}")

        self.counter += 1
        self.current_group.create_dataset(name, data=data, compression="lzf", **kwargs)
        return None

    def close(self):
        self.file.close()
        return None


class DataSet(torch.utils.data.Dataset):
    def __init__(self, *, filename, percentage, augment=True):
        self.bins = -1
        self.filename = filename 
        self.regex = re.compile('_class(\d{1,})\.jpg')
        self.augment = augment 
        
        patch_names = []
        with h5py.File(f"{processed_data_folder}/{self.filename}", "r") as filein:
            names = list(filein.keys())
            # this is for nested h5files, writing nested h5file is much more efficient
            for key in names:
                if isinstance(filein[key], h5py._hl.group.Group):
                    dataset_keys = list(filein[key].keys())
                    patch_names.extend([f"{key}/{sub_key}" for sub_key in dataset_keys])
                else:
                    patch_names.append(key)
        start, end = percentage
        self.names = patch_names[int(len(patch_names)*start) : int(len(patch_names)*end)]
        self.std = np.std(list(range(256)))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        data_name = self.names[index]
        with h5py.File(f"{processed_data_folder}/{self.filename}", "r") as filein:
            input_data = filein[data_name][:]

        input_data = (input_data - 127.5) / self.std
        
        if self.augment:
            input_data = self.augmenter.augment(input_data)
        
        image_class = int(self.regex.search(data_name).groups()[0])
        return torch.from_numpy(input_data), image_class


def organize(): 

    def concat_df(main_table, results): 
        result_df = pd.DataFrame() 
        result_df['class_name'] = results['class_name']
        result_df['total_samples'] = results['total_samples']
        result_df['dataset'] = results['dataset']    
        result_df['id'] = results['id']    
        
        result_df = result_df.sort_values(by=['class_name'])
        return pd.concat([main_table, result_df], axis=0).reset_index(drop=True)

    # FoodRecChallenge
    def FoodRecChallenge():
        current_dataset = dict() 

        for sub_folder in ['train', 'val']:     
            with open(f'{main_path}/FoodRecChallenge/{sub_folder}/annotations.json') as filein: 
                current = json.load(filein)
            for each_item in current['annotations']: 
                if each_item['category_id'] not in current_dataset.keys():
                    current_dataset[each_item['category_id']] = {'count': 1}
                else:
                    current_dataset[each_item['category_id']]['count'] += 1 

        with open(f'{main_path}/FoodRecChallenge/{sub_folder}/annotations.json') as filein: 
            current = json.load(filein)

        for each_item in current['categories']: 
            if each_item['id'] not in current_dataset.keys():
                print(f'error, {each_item["id"]}')
            else:
                current_dataset[each_item['id']]['name'] = each_item['name']
                current_dataset[each_item['id']]['name_readable'] = each_item['name_readable']

        name_list, samples_list, id_list = [], [], []
        # current_np = np.ndarray(shape=(len(current_dataset), embed_size))
        for _, (each_id, package) in enumerate(current_dataset.items()): 
            name_list.append(package['name'])
            samples_list.append(package['count'])
            id_list.append(each_id)
            # current_np[index] = get_embedding(package['name_readable'])            
        return {'class_name': name_list, 'total_samples': samples_list, 'dataset': 'FoodRecChallenge', 'id': id_list}

    # ISIAFood_500
    def ISIAFood_500(): 
        result = {'class_name': [], 'total_samples': [], 'dataset': 'ISIAFood_500', 'embed': 0}
        path = f'{main_path}/ISIAFood_500/dataset/images'
        
        for _, each_folder in enumerate(os.listdir(path)): 
            
            result['class_name'].append(each_folder)
            result['total_samples'].append(len(f'{path}/{each_folder}'))
        
        result['id'] = -1
        return result

    # UECFood_256
    def UECFood_256(): 
        result = {'class_name': [], 'total_samples': [], 'dataset': 'UECFood_256', 'id': []}
        path = f'{main_path}/UECFood_256/images'
        
        with open(f'{path}/category.txt', 'r') as filein:
            data = filein.read().strip().split('\n')[1:]
            data = [item.split('\t', 2) for item in data]
            cat_mapping = {key: value for key, value in data}
            
        for _, folder in enumerate(filter(lambda each: os.path.isdir(f'{path}/{each}'), os.listdir(path))):
            result['id'].append(int(folder))
            result['class_name'].append(cat_mapping[folder])
            result['total_samples'].append(len(os.listdir(f'{path}/{folder}')))
            
        return result 

    # # food_101
    # def food_101(): 
    #     result = {'class_name': [], 'total_samples': [], 'dataset': 'food_101', 'embed': 0}
    #     path = f'{main_path}/food_101/images'
            
    #     for _, folder in enumerate(filter(lambda each: os.path.isdir(f'{path}/{each}'), os.listdir(path))):
    #         result['class_name'].append(folder)
    #         result['total_samples'].append(len(os.listdir(f'{path}/{folder}')))
            
    #     result['id'] = 0
    #     return result 

    # # ifood_251
    # def ifood_251(): 
    #     result = {'class_name': [], 'total_samples': [], 'dataset': 'ifood_251', 'id': []}
    #     result = {} 
    #     path = f'{main_path}/ifood_251'
        
    #     with open(f'{path}/class_list.txt', 'r') as filein: 
    #         data = filein.read().strip().split('\n')
    #         data = [item.split(maxsplit=2) for item in data]
    #         result = {int(key): {'name': value, 'count': 0} for key, value in data}
            
    #     for case in ['train', 'val']: 
    #         with open(f'{path}/{case}_labels.csv', 'r') as filein: 
    #             data = filein.read().strip().split('\n')[1:]
    #             data = [item.split(',')[1] for item in data] 
    #         for each in set(data): 
    #             result[int(each)]['count'] += data.count(each)
            
    #     name_list, samples_list, id_list = [], [], []
    #     for _, (item_id, package) in enumerate(result.items()): 
    #         id_list.append(int(item_id))
    #         name_list.append(package['name'])
    #         samples_list.append(package['count'])
            
    #     return {'class_name': name_list, 'total_samples': samples_list, 'dataset': 'ifood_251', 'id': id_list} 

    header = '/home/michael/SSD_Cache'
    main_path = f'{header}/PEAR/raw_data'

    result_table = pd.DataFrame(columns=['class_name', 'total_samples', 'dataset', 'id'])
    result_table = result_table.iloc[:1000]
    
    for func in tqdm([FoodRecChallenge, ISIAFood_500, UECFood_256]):
        result_table = concat_df(result_table, func())
    
    raw_path = f'{header}/PEAR/raw_data'
    output_dir = f'{header}/PEAR/organized_raw'
    organized_data_counter = count(0)

    rmtree(output_dir)
    os.mkdir(output_dir)
    
    # FoodRecChallenge
    for d_type in ['train', 'val']: 
        folder_path = f'{raw_path}/FoodRecChallenge/{d_type}/images'
        dataset_record = json.load(open(f'{raw_path}/FoodRecChallenge/{d_type}/annotations.json'))
        for each_record in tqdm(dataset_record['annotations'], ncols=80, desc='FoodRecChallenge'):
            class_id = each_record['category_id']
            item_class = result_table.loc[(result_table['dataset']=='FoodRecChallenge') & 
                                        (result_table['id'] == class_id)].index[0]        
            item_id = each_record['image_id']
            output_name = f'{next(organized_data_counter):>06}_class{item_class}.jpg'
            copyfile(f'{folder_path}/{item_id:>06}.jpg', f'{output_dir}/{output_name}')

    # ISIAFood_500
    folder_path = f'{raw_path}/ISIAFood_500/dataset/images'
    for folder in tqdm(os.listdir(folder_path), ncols=80, desc='ISIAFood_500'): 
        class_id = result_table.loc[(result_table['dataset']=='ISIAFood_500') & 
                                    (result_table['class_name'] == folder)].index[0] 
        for each_item in os.listdir(f'{folder_path}/{folder}'):
            output_name = f'{next(organized_data_counter):>06}_class{class_id}.jpg'
            copyfile(f'{folder_path}/{folder}/{each_item}', f'{output_dir}/{output_name}')
        
    # UECFood_256
    folder_path = f'{raw_path}/UECFood_256/images'
    for folder in tqdm(os.listdir(folder_path), ncols=80, desc='UECFood_256'): 
        try:
            class_id = result_table.loc[(result_table['dataset']=='UECFood_256') & 
                                        (result_table['id'] == int(folder))].index[0] 
            for each_item in os.listdir(f'{folder_path}/{folder}'):
                output_name = f'{next(organized_data_counter):>06}_class{class_id}.jpg'
                copyfile(f'{folder_path}/{folder}/{each_item}', f'{output_dir}/{output_name}')
        except:
            continue 


def make_model_data():

    tool = PatchAugmentor()
    all_data = os.listdir(raw_data_folder)
    labeled_h5 = FastH5(h5py.File(f"{processed_data_folder}/labeled_train.h5", "w"), len(all_data))
    shuffle(all_data)
    failed = 0 
    for filename in tqdm(all_data): 
        try:
            image = cv2.imread(f'{raw_data_folder}/{filename}')
            image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
            labeled_h5.create_dataset(filename, tool.to_3xy(image))
        except:
            failed += 1 
    print(f'{failed}/{len(all_data)}')
    print("data processing complete")
    return None

    
def mask_image_raw(image, mask): 
    width = image.shape[1] // mask.shape[1]
    
    print(width)
    for x_index in range(mask.shape[1]): 
        for y_index in range(mask.shape[2]): 
            image[: , x_index*width:(x_index+1)*width , y_index*width:(y_index+1)*width] *= mask[: , x_index , y_index][0]
    return image


class PatchAugmentor:
    
    def __init__(self):
        self.mask_image = np.vectorize(mask_image_raw, cache=True)
    
    def to_3xy(self, image): 
        # xy3 to 3xy
        data = np.zeros((3, image.shape[0], image.shape[1]), dtype=image.dtype)
        data[0] = image[:,:,0]
        data[1] = image[:,:,1]
        data[2] = image[:,:,2]
        return data 

    def to_xy3(self, image):
        # 3xy to xy3 
        data = np.zeros((image.shape[1], image.shape[2], 3), dtype=image.dtype)
        data[:,:,0] = image[0]
        data[:,:,1] = image[1]
        data[:,:,2] = image[2]
        return data 

    def augment(self, image):
        
        if randint(0, 1): 
            image = np.flipud(image)
        
        if randint(0, 1): 
            image = np.fliplr(image)
            
        if randint(0, 1): 
            image = np.transpose(image, axes=(0, 2, 1))
            
        rot_mode = randint(0, 3)
        if rot_mode == 0: 
            image = np.rot90(image)
        elif rot_mode == 1: 
            image = np.rot90(image)
            image = np.rot90(image)
        elif rot_mode == 2: 
            image = np.rot90(image)
            image = np.rot90(image)
            image = np.rot90(image)
        
        mask_mode = randint(0, 2)
        if mask_mode == 0: 
            x_index = randint(0, image.shape[0]-32)
            y_index = randint(0, image.shape[1]-32)
            image[:, x_index:x_index+32, y_index:y_index+32] = 0 
        elif mask_mode == 1: 
            percentage = randint(0, 10)
            mask = np.random.randint(low=0, high=10, size=(3, 8, 8, 1)) >= percentage
            image = self.mask_image(image, mask)
        
        pad_mode = randint(0, 2)
        old_size = image.shape[1:]
        width = (randint(0, 5) / 10) * image.shape[0]
        if pad_mode == 0: 
            image = np.pad(image, 
                        (
                            (0,0), 
                            (int(width*image.shape[1]), int(width*image.shape[1])),
                            (0,0)
                            )
                        )
            image = cv2.resize(self.to_xy3(image), old_size, interpolation=cv2.INTER_CUBIC)
            image = self.to_3xy(image)
        elif pad_mode == 1: 
            image = np.pad(image, 
                        (
                            (0,0), 
                            (0,0)
                            (int(width*image.shape[1]), int(width*image.shape[1])),
                            )
                        )
            image = cv2.resize(self.to_xy3(image), old_size, interpolation=cv2.INTER_CUBIC)
            image = self.to_3xy(image)
        
        return image


if __name__ == '__main__':
    organize() 