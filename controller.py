from PIL import Image
from model.extractor import FeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from model.load import loader, loader2



def main(my_data):
    # Insert the image query
    img = Image.open(my_data)
    # Extract its features
    fe = FeatureExtractor()
    query = fe.extract(img)

    # Calculate the similarity (distance) between images
    #img_paths, features = loader()
    img_paths, features = loader2()
    dists = np.linalg.norm(features - query, axis=1)

    # Extract 30 images that have lowest distance
    ids = np.argsort(dists)[:30]

    scores = [(dists[id], img_paths[id], id) for id in ids]
    # Visualize the result
    axes=[]
    fig=plt.figure(figsize=(4,4))
    for a in range(2*3):
        score = scores[a]
        axes.append(fig.add_subplot(2, 3, a+1))
        character = score[1].split('/')[-1]  # 파일 경로에서 파일 이름 추출
        character = character[:-4]
        subplot_title=str(round(score[0],2)) + "//" + character
        plt.axis('off')
        plt.imshow(Image.open(score[1]))
        plt.title(subplot_title)
    fig.tight_layout()
    plt.show()

    return img
