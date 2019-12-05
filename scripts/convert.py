import os
import cv2
input_path = './noise'
output_path = ''

for file_name in os.listdir(input_path):
    file_path = os.path.join(input_path, file_name)
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_path, file_name), image)

