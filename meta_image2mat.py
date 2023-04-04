import cv2
import numpy
import meta_dataset


class Image2Mat:
    def __init__(self, image_path_list, image_width, image_height):
        self.img_mat = None

        self.image_width = image_width
        self.image_height = image_height
        self.image_path_list = image_path_list

        self.image2mat()

    def image2mat(self):
        count = 0
        counter = len(self.image_path_list)

        img_size = self.image_width * self.image_height
        img_mat = numpy.zeros((img_size, counter))
        for img in self.image_path_list:
            # 当第二个参数为0时，cv2.imread()函数会将图像以灰度模式读入，即将彩色图像转换为灰度图像。
            gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            # resize to width * height specific
            gray_resized = cv2.resize(gray, (self.image_width, self.image_height))
            # 将图像转换为一维数组
            vec_mat = numpy.asmatrix(gray_resized).ravel()
            img_mat[:, count] = vec_mat
            count += 1

        self.img_mat = img_mat
        self.print_out_for_check_purpose()

        return img_mat

    def print_out_for_check_purpose(self):
        print("\n---------------------------------")
        print(f"Image Width: {self.image_width}")
        print(f"Image Height: {self.image_height}")
        print(f"Image Size: {self.image_width * self.image_height}")

        print(f"Image Mat: \n{self.img_mat}")
        print(f"Image Mat Shape: {self.img_mat.shape}")

    def get_image_mat(self):
        return self.img_mat

    def get_image_path_list(self):
        return self.image_path_list

    def get_image_width(self):
        return self.image_width

    def get_image_height(self):
        return self.image_height


if __name__ == "__main__":
    datasetClass = meta_dataset.DatasetClass('image_train')
    image_mat = Image2Mat(datasetClass.get_training_image_path(), 100, 100)
