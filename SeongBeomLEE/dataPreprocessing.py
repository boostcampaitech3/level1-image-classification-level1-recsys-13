import os
import pandas as pd

class dataPreProcessing:
    def __init__(self, df: pd.DataFrame, swap_mask_li: list, swap_gender_li: list):
        self.df = df.copy()
        self.swap_mask_li = swap_mask_li
        self.swap_gender_li = swap_gender_li

    def make_pre_df(self) -> pd.DataFrame:
        preprocessing_df = self.swap_gender(swap_li=self.swap_gender_li, df=self.df)
        preprocessing_df['ages'] = preprocessing_df['age'].apply(lambda x: self.get_ages(x))
        preprocessing_df['genders'] = preprocessing_df['gender'].apply(lambda x: self.get_genders(x))
        preprocessing_df['cv_taget_col'] = 'age' + '_' + preprocessing_df['age'].astype(str) + '_' + 'genders' + '_' + \
                                           preprocessing_df['genders'].astype(str)

        return preprocessing_df

    def make_train_df(self, df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
        train_df = []

        for line in df.iloc:
            for file in list(os.listdir(os.path.join(image_dir, line['path']))):
                if file[0] == '.':
                    continue

                mask = file.split('.')[0]
                gender = line['gender']
                age = line['age']

                masks = self.get_masks(mask)
                genders = self.get_genders(gender)
                ages = self.get_ages(age)
                age_cats = self.get_age_cats(age)

                data = {
                    'id': line['id'],
                    'mask': mask,
                    'gender': gender,
                    'age': age,
                    'masks': masks,
                    'genders': genders,
                    'ages': ages,
                    'age_cats': age_cats,
                    'labels': self.get_labels(masks=masks, genders=genders, ages=ages),
                    'label_cats': self.get_label_cats(masks=masks, genders=genders, ages=age_cats),
                    'path': os.path.join(image_dir, line['path'], file),
                }

                train_df.append(data)

        train_df = pd.DataFrame(train_df)

        train_df['idx'] = train_df.index

        train_df = self.swap_mask(swap_li=self.swap_mask_li, df=train_df)

        return train_df

    def get_ages(self, x) -> int:
        if x < 30:
            return 0
        elif x < 60:
            return 1
        else:
            return 2

    def get_genders(self, x) -> int:
        if x == 'male':
            return 0
        else:
            return 1

    def get_masks(self, x) -> int:
        if x == 'normal':
            return 2
        elif x == 'incorrect_mask':
            return 1
        else:
            return 0

    def get_age_cats(self, x) -> int:
        if x < 25:
            return 0
        elif x < 30:
            return 1
        elif x < 45:
            return 2
        elif x < 52:
            return 3
        elif x < 57:
            return 4
        elif x < 60:
            return 5
        else:
            return 6

    def get_labels(self, masks, genders, ages) -> int:
        return masks * 6 + genders * 3 + ages

    def get_label_cats(self, masks, genders, ages) -> int:
        return masks * 12 + genders * 6 + ages

    def swap_gender(self, swap_li: list, df: pd.DataFrame) -> pd.DataFrame:
        swap_df = df.copy()
        if swap_li:
            for swap in swap_li:
                swap_id, swap_gender = swap
                swap_df.loc[swap_df[swap_df['id'] == swap_id].index, 'gender'] = swap_gender
        return swap_df

    def swap_mask(self, swap_li: list, df: pd.DataFrame) -> pd.DataFrame:
        swap_df = df.copy()
        if swap_li:
            for swap_id in swap_li:
                _swap_df = swap_df[swap_df['id'] == swap_id]

                normal_swap_df = _swap_df[_swap_df['mask'] == 'normal']
                incorrect_mask_swap_df = _swap_df[_swap_df['mask'] == 'incorrect_mask']

                normal_path = normal_swap_df['path'].values[0]
                incorrect_mask_path = incorrect_mask_swap_df['path'].values[0]

                swap_df.loc[normal_swap_df.index, 'path'] = incorrect_mask_path
                swap_df.loc[incorrect_mask_swap_df.index, 'path'] = normal_path

        return swap_df