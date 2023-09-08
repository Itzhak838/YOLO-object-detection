import cv2 as cv
from ultralytics import YOLO
import math
from skimage.transform import resize
import skimage
"""
the following code is used to detect a big data of images 
its detect the images and save the data to a file by name, confidence and compression ratio
in this case, the images are in format: c/d_b/g/r_1-10.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "p_data.txt"
"""

l1 = ["c", "d"]  # c: cat, d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: gray, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # compression ratio loop
for i in l1:
    for j in l2:
        for k in l3:
            for l in l4:
                path = "inference/images/data_all/" + i + "_" + j + "_" + str(k) + ".jpg"  # build the path to the image by lists
                image_input = cv.imread(path)  # read the image by path
                pixels_2_one_pixel = l  # compression ratio
                scale = 1/math.sqrt(pixels_2_one_pixel)  # calculate the scale factor to resize the image by the compression ratio
                resized_image = resize(image_input, (image_input.shape[0] * scale, image_input.shape[1] * scale), mode='constant', anti_aliasing=False)  # resize the image by the scale factor
                rescaled_image = resize(resized_image, (image_input.shape[0], image_input.shape[1]), anti_aliasing=False)  # rescale the image to the original size
                rescaled_image = skimage.img_as_ubyte(rescaled_image)  # convert the image to uint8
                cv.imwrite("inference/images/resized.jpg", rescaled_image)  # save the image to print it later

                # load a pretrained YOLOv8n model
                model = YOLO("yolov8n.pt", "v8")

                # predict the compressed image
                detection_output = model.predict(rescaled_image, conf=0.05, save=False)

                # Display tensor array
                # print(detection_output)

                # Display numpy array
                # print(detection_output[0].numpy())

                # Loop through the detection output to save the data to a file by name and confidence
                # the data is saved in the file "p_data.txt"
                for r in detection_output:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]  # get the coordinates of the bounding box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # print(x1,y1,x2,y2)
                        conf = box.conf.numpy()[0]  # get the probability of the bounding box
                        name = int(box.cls.numpy()[0])  # get the class number of the bounding box

                        # if the predicted is P( cat | cat ) save the data to the file by name, confidence and compression ratio
                        if name == 15:
                            if i == "c":
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(l) + " cat " + str(conf) + "\n"
                                with open("p_data.txt", "a") as file:
                                    file.write(data_2_file)
                        # if the predicted is P( dog | dog ) save the data to the file by name, confidence and compression ratio
                        if name == 16:
                            if i == "d":
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(l) + " dog " + str(conf) + "\n"
                                with open("p_data.txt", "a") as file:
                                    file.write(data_2_file)

