# PCA Face Recognition

import numpy
import cv2 as cv

import meta_dataset


class PCAFace2Matrix:
    def __init__(self, images, image_width, image_height):
        self.img_matrix = None
        self.images_path = images
        self.image_width = image_width
        self.image_height = image_height
        self.image_size = image_width * image_height

    def image_matrix_transformation(self):
        counter = 0
        count = len(self.images_path)
        img_matrix = numpy.zeros((self.image_size, count))
        for image in self.images_path:
            gray = cv.imread(image, 0)
            gray_resized = cv.resize(gray, (self.image_width, self.image_height))
            matrix_gray = numpy.asmatrix(gray_resized)

            # to ravel the matrxi
            vec = matrix_gray.ravel()
            img_matrix[:, counter] = vec
            counter += 1
        self.img_matrix = img_matrix
        return img_matrix

    def print_for_check_purpose(self):
        print("\n---------------------------------")
        print("image width: ", self.image_width)
        print("image height: ", self.image_height)
        print("image size: ", self.image_size)
        print("image matrix: \n", self.img_matrix)
        print("image matrix shape: ", self.img_matrix.shape)
        print("images path: ", self.images_path)

    def get_image_matrix(self):
        return self.img_matrix

    def get_image_path(self):
        return self.images_path

    def get_image_width(self):
        return self.image_width

    def get_image_height(self):
        return self.image_height


if __name__ == "__main__":
    datasetClass = meta_dataset.DatasetClass('../image_train')
    image2matrix_obj = PCAFace2Matrix(datasetClass.get_training_image_path(), 100, 100)
    image2matrix_obj.image_matrix_transformation()
    image2matrix_obj.print_for_check_purpose()

