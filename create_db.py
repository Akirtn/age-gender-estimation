import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta
from os import listdir
from os.path import isfile, join
import sys
import dlib
from moviepy.editor import *

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb or asia")
                        #I add asia version
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
                        #img_size was changed default is 32.
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    #output_path = args.output
    output_path = './data/megaage_test'
    #output_path = './data/megaage_train'
    # traning or test .mat is generated under data folder
    db = args.db
    img_size = args.img_size
    min_score = args.min_score
    mypath = './megaage_asian/test'
    #mypath = './megaage_asian/train'
    isPlot = False
    #learning data megaasian folder place
    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)

    age_file = np.loadtxt('./megaage_asian/list/test_age.txt')
    #age_file = np.loadtxt('./megaage_asian/list/train_age.txt')
    img_name_file = np.genfromtxt('./megaage_asian/list/test_name.txt',dtype='str')
    #img_name_file = np.genfromtxt('./megaage_asian/list/train_name.txt',dtype='str')
    gender_file =np.loadtxt('./megaage_asian/list/test_gender.txt')
    #gender_file =np.loadtxt('./megaage_asian/list/train_gender.txt')
    #full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []
    for i in tqdm(range(len(img_name_file))):
    #for i in tqdm(range(len(face_score))):
        # if face_score[i] < min_score:
        #     continue
        # if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
        #     continue
        # #isnan is missing value　completion
        # if ~(0 <= age[i] <= 100):
        #     continue

        if np.isnan(gender[i]):
            continue

        # out_genders.append(int(gender[i]))
        # out_ages.append(age[i])
        # img = cv2.imread(root_path + str(full_path[i][0]))
        # out_imgs.append(cv2.resize(img, (img_size, img_size)))
        input_img = cv2.imread(mypath+'/'+img_name_file[i])
        input_img = input_img[20:-20,:,:]
        #行（height） x 列（width） x 色（color）の三次元のndarray
        img_h, img_w, _ = np.shape(input_img)
        age = int(float(age_file[i]))
        if age >= -1:
	        if isPlot:
		        img_clip = ImageClip(input_img)
		        img_clip.show()
		        key = cv2.waitKey(1000)

	        input_img = cv2.resize(input_img,(img_size,img_size))
            #only add to the list when faces is detected
            out_imgs.append(input_img)       
	        out_ages.append(age[i])
            out_genders.append(int(gender[i]))
    output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
