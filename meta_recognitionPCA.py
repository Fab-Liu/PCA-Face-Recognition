import numpy
from PIL import Image

import meta_dataset
import meta_image2mat


class RecognitionPCA:
    def __init__(self, image_matrix, image_label, image_width, image_height, category_name, category_num):
        self.new_bases = None
        self.mean_face = None
        self.new_coordinates = None
        self.reduced_image_matrix = None

        self.quality = 90
        self.image_matrix = image_matrix

        self.image_label = image_label
        self.image_width = image_width
        self.image_height = image_height

        self.category_name = category_name
        self.category_num = category_num

    def calculate_mean_face(self):
        mean_face = numpy.mean(self.image_matrix, axis=1)
        print("Mean: ", mean_face.shape)
        self.mean_face = numpy.asmatrix(mean_face).T
        print("Mean_face: ", self.mean_face.shape)
        self.reduced_image_matrix = self.image_matrix - self.mean_face
        return self.mean_face, self.reduced_image_matrix

    # suppose p is the number of eigenvalues that are greater than the threshold
    def calculate_p_value(self, eig_vals):
        total = numpy.sum(eig_vals)
        threshold = total * self.quality / 100

        p_value = 0
        temp_sum = 0
        while temp_sum < threshold:
            temp_sum += eig_vals[p_value]
            p_value += 1
        return p_value

    def reduce_lim(self):
        e, eig_vals, v_t = numpy.linalg.svd(self.image_matrix, full_matrices=True)
        p_value = self.calculate_p_value(eig_vals)
        self.new_bases = e[:, 0:p_value]
        self.new_coordinates = numpy.dot(self.new_bases.T, self.reduced_image_matrix)
        return self.new_coordinates

    def new_coor(self, recognize_image):
        img_vec = numpy.asmatrix(recognize_image).ravel().T
        new_mean = (self.mean_face * len(self.image_label) + img_vec) / (len(self.image_label) + 1)
        img_vec = img_vec - new_mean
        return numpy.dot(self.new_bases.T, img_vec)

    def recognition(self, coordinates):
        classes = len(self.category_num)
        count = 0
        dist = []
        for i in range(classes):
            temp_img = self.new_coordinates[:, int(count):int(count + self.category_num[i])]
            mean_temp = numpy.asmatrix(numpy.mean(temp_img, axis=1))
            count = count + self.category_num[i]
            dis_temp = numpy.linalg.norm(coordinates - mean_temp)
            dist += [dis_temp]
        min_pos = numpy.argmin(dist)
        print("The distance is: ", dist[min_pos])
        if dist[min_pos] > 5000:
            pass
        else:
            print("The image is: ", self.category_name[min_pos])
            return self.category_name[min_pos]


if __name__ == "__main__":
    datasetClass = meta_dataset.DatasetClass('image_train')
    image_mat = meta_image2mat.Image2Mat(datasetClass.get_training_image_path(), 100, 100)

    pca_matrix = image_mat.get_image_mat()
    image_label = datasetClass.get_training_label()
    image_width = image_mat.get_image_width()
    image_height = image_mat.get_image_height()
    category_name = datasetClass.get_category_name()
    category_num = datasetClass.get_category_num()

    recognitionPCA = RecognitionPCA(pca_matrix, image_label, image_width, image_height, category_name, category_num)
    recognitionPCA.calculate_mean_face()
    recognitionPCA.reduce_lim()

    # # read image in array
    # image_path = "./image_recognize/face/20.jpg"
    # recognize_image = numpy.array(Image.open(image_path).convert('L'))
    #
    # recognitionPCA.new_coor(recognize_image)
    # print(recognitionPCA.recognition(recognitionPCA.new_coor(recognize_image)))
