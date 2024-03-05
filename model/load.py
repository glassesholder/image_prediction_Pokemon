from settings import root_dir
from model.extractor import FeatureExtractor
import os
from PIL import Image
import numpy as np

File_names = os.listdir(root_dir)

features = []
img_paths = []

# Save Image Feature Vector with Database Images
def loader():
    for i in File_names:
        try:
            image_path = "./data/images/" + str(i)
            img_paths.append(image_path)

            # Extract Features
            fe = FeatureExtractor()
            feature = fe.extract(img=Image.open(image_path))

            features.append(feature)

            # Save the Numpy array (.npy) on designated path
            feature_path = "./data/images/" + str(i)[:-4] + ".npy"
            np.save(feature_path, feature)
        except Exception as e:
            print('예외가 발생했습니다.', e)

    return img_paths, features


def loader2():
    for i in File_names:
        try:
            # 확장자가 .npy인지 확인
            if i.endswith(".jpg") or i.endswith(".png"):
                # npy 파일 경로 설정
                npy_path = "./data/images/" + str(i)[:-4] + ".npy"
                
                # npy 파일로부터 특성 불러오기
                feature = np.load(npy_path)
                
                # 특성과 이미지 경로를 리스트에 추가
                features.append(feature)
                image_path = "./data/images/" + str(i)
                img_paths.append(image_path)
                
        except Exception as e:
            print('예외가 발생했습니다.', e)
    
    return img_paths, features