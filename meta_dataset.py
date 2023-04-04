import os

import meta_preprocessing


class DatasetClass:
    def __init__(self, dataset_path):
        # dataset name
        self.dataset_path = dataset_path

        self.training_image_path = []
        self.training_label = []

        self.category_name = []
        self.category_num = []

        self.load_dataset_from_dir()

    def load_dataset_from_dir(self):
        # os check exist
        if not os.path.exists(self.dataset_path):
            raise ValueError("dataset_dir not exists")
        else:
            print("Dataset Establish : dataset_dir detected.")

        print(f"Dataset Path: {os.listdir(self.dataset_path)}")
        print(f"Dataset Path Num: {len(os.listdir(self.dataset_path))}")
        print("\n---------------------------------")

        for category in os.listdir(self.dataset_path):
            # os check exist
            if not os.path.exists(os.path.join(self.dataset_path, category)):
                raise ValueError("category_dir not exists")
            else:
                print(f"Category Establish : {category} detected.")

            if category == '.DS_Store':
                continue
            else:
                # category name
                self.category_name.append(category)

                # category num
                self.category_num.append(len(os.listdir(os.path.join(self.dataset_path, category))))

                # training image path
                for image in os.listdir(os.path.join(self.dataset_path, category)):
                    if image == '.DS_Store':
                        continue
                    else:
                        print(f"Image Path: {os.path.join(self.dataset_path, category, image)}")
                        print(f"Image Shape: {meta_preprocessing.load_img(os.path.join(self.dataset_path, category, image)).shape}")
                        # img = meta_preprocessing.load_img(os.path.join(self.dataset_path, category, image))
                        # #如果不是灰度图，转换为灰度图
                        # if len(img.shape) == 3:
                        #     img = meta_preprocessing.rgb2gray(img)
                        self.training_image_path.append(os.path.join(self.dataset_path, category, image))

                # training label
                self.training_label.append([0] * len(os.listdir(os.path.join(self.dataset_path, category))))
        self.print_out_for_check_purpose()

    def print_out_for_check_purpose(self):
        print("\n---------------------------------")
        print(f"Category Name: {self.category_name}")
        print(f"Category Name Num: {len(self.category_name)}")
        print(f"Category Num: {self.category_num}")
        print(f"Category Num: {len(self.category_num)}")
        print(f"Training Image Path: {self.training_image_path}")
        print(f"Training Image Path Num: {len(self.training_image_path)}")
        print(f"Training Label: {self.training_label}")
        print(f"Training Label Num: {len(self.training_label)}")

    def get_training_image_path(self):
        return self.training_image_path

    def get_training_label(self):
        return self.training_label

    def get_category_name(self):
        return self.category_name

    def get_category_num(self):
        return self.category_num


if __name__ == "__main__":
    datasetClass = DatasetClass('image_train')
