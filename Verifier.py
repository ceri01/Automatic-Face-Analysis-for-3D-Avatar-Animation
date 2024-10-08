import subprocess
import time
import numpy as np
import scipy.stats as st
import scipy.spatial.distance as dist
import os

from openFACS.python.openFACS import sendAUS
from TestUtils import catch_aus_libreface, catch_aus_feat
from feat import Detector
from Screenshot_performer import perform

datasets = ["dataset/AffectNet/", "dataset/IAS-lab/"]
avatar_image_location = "avatar-screen/"
subprocess.Popen(["LinuxNoEditor/ActionUnitsFace.sh"])
time.sleep(15)
results_feat = []
results_libreface = []
results_between_models = []
times_libreface = []
times_feat = []

comparisons = []

def run():
    avatar_image_number = 0
    detector = Detector(face_model="retinaface", landmark_model="mobilefacenet", au_model='xgb')

    for dataset in datasets:
        for filename in os.listdir(dataset):
            start = time.time()
            img_aus_libreface = catch_aus_libreface(dataset + filename)
            end = time.time()
            times_libreface.append(end - start)

            start = time.time()
            img_aus_feet = catch_aus_feat(dataset + filename, detector)
            end = time.time()
            times_feat.append(end - start)

            if len(img_aus_libreface) <= 0:
                continue

            sendAUS(img_aus_libreface, 1)
            time.sleep(0.5)
            perform(avatar_image_location + str(avatar_image_number) + ".png")  # perform screenshot
            avatar_aus_libreface = catch_aus_libreface(avatar_image_location + str(avatar_image_number) + ".png")
            avatar_aus_feat = catch_aus_feat(avatar_image_location + str(avatar_image_number) + ".png", detector)

            avatar_image_number += 1

            if len(avatar_aus_libreface or avatar_aus_feat) <= 0:
                continue

            # get cosine similarity between image aus and avatar aus (1 equals 0 completely different)
            results_between_models.append(1 - dist.cosine(img_aus_libreface, img_aus_feet))

            results_feat.append(1 - dist.cosine(img_aus_feet, avatar_aus_feat))
            results_libreface.append(1 - dist.cosine(img_aus_libreface, avatar_aus_libreface))

    print("results of comparison between AUs caught from images and avatar using pyfeat")
    make_results(results_feat)

    print("results of comparison between AUs caught from images and avatar using libreface")
    make_results(results_libreface)

    print("results of comparison between AUs models")
    make_results(results_between_models)

    feat_time_mean = np.mean(times_feat)
    libreface_time_mean = np.mean(times_libreface)
    print(f"media tempo py-feat{feat_time_mean}")
    print(f"media tempo libreface {libreface_time_mean}")


def make_results(final_results, confidence=0.95):
    dataset_size = len(final_results)
    aus_mean = np.mean(final_results)
    aus_dev_std = np.std(final_results)
    print(f"media AUs {aus_mean}")
    print(f"dev std AUs {aus_dev_std}")

    std_error = st.sem(final_results)  # Errore standard della media
    h = std_error * st.t.ppf((1 + confidence) / 2., dataset_size - 1)  # Calcola l'ampiezza dell'intervallo di confidenza

    lower_bound = aus_mean - h
    upper_bound = aus_mean + h
    print(f"Intervallo di confidenza ({confidence * 100}%): ({lower_bound}, {upper_bound})")

run()
