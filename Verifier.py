import subprocess
import time
import numpy as np
import scipy.stats as st
import scipy.spatial.distance as dist
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from openFACS.python.openFACS import sendAUS
from TestUtils import catch_aus_libreface, catch_aus_feat
from feat import Detector
from Screenshot_performer import perform

datasets = ["dataset/AffectNet/", "dataset/IAS-lab/"]
avatar_image_location = "avatar-screen-screen/"
subprocess.Popen(["LinuxNoEditor/ActionUnitsFace.sh"])
time.sleep(15)

au_labels = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU25", "AU26", "AU28", "AU43"
]

img_aus = []
avatar_aus = []

results_feat_cosim = []
results_libreface_consim = []
results_between_models_cosim = []

results_feat_pearson = []
results_libreface_pearson = []
results_between_models_pearson = []

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
            img_aus.append(img_aus_libreface)
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
            avatar_aus.append(avatar_aus_libreface)
            avatar_aus_feat = catch_aus_feat(avatar_image_location + str(avatar_image_number) + ".png", detector)

            avatar_image_number += 1

            if len(avatar_aus_libreface or avatar_aus_feat) <= 0:
                continue

            # get cosine similarity between image aus and avatar aus (1 equals 0 completely different)
            results_between_models_cosim.append(1 - dist.cosine(img_aus_libreface, img_aus_feet))
            coef_mod, p_mod = st.pearsonr(img_aus_libreface, img_aus_feet)
            if p_mod < 0.05:
                results_between_models_pearson.append(coef_mod)

            results_feat_cosim.append(1 - dist.cosine(img_aus_feet, avatar_aus_feat))
            coef_feat, p_feat = st.pearsonr(img_aus_feet, avatar_aus_feat)
            if p_feat < 0.05:
                results_feat_pearson.append(coef_feat)

            results_libreface_consim.append(1 - dist.cosine(img_aus_libreface, avatar_aus_libreface))
            coef_lib, p_lib = st.pearsonr(img_aus_libreface, avatar_aus_libreface)
            if p_lib < 0.05:
                results_libreface_pearson.append(coef_lib)

    print("results of comparison between AUs caught from images and avatar using pyfeat and cosine similarity\n")
    make_results(results_feat_cosim)
    print("results of comparison between AUs caught from images and avatar using pyfeat and pearson index\n")
    make_results(results_feat_pearson)


    print("results of comparison between AUs caught from images and avatar using libreface and cosine similarity\n")
    make_results(results_libreface_consim)
    print("results of comparison between AUs caught from images and avatar using libreface and pearson index\n")
    make_results(results_libreface_pearson)

    print("results of comparison between AUs models using cosine similarity\n")
    make_results(results_between_models_cosim)
    print("results of comparison between AUs models using pearson index\n")
    make_results(results_between_models_pearson)
    print("\n")

    feat_time_mean = np.mean(times_feat)
    libreface_time_mean = np.mean(times_libreface)
    print(f"media tempo py-feat{feat_time_mean}")
    print(f"media tempo libreface {libreface_time_mean}")

    # print grafici

    plot_au_difference_heatmap(img_aus, avatar_aus, au_labels)
    plot_au_scatter_plots(img_aus, avatar_aus, au_labels)
    plot_pearson_correlation_boxplot(results_libreface_pearson)



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


# Funzione per plottare la Heatmap delle Differenze tra le Action Units
def plot_au_difference_heatmap(au_real, au_avatar, au_names):
    """
    Plotta una heatmap delle differenze assolute tra le Action Units.

    Parametri:
    - au_real: array 2D o DataFrame delle AUs delle immagini reali (righe: immagini, colonne: AUs).
    - au_avatar: array 2D o DataFrame delle AUs dell'avatar.
    - au_names: lista dei nomi delle Action Units corrispondenti alle colonne.
    """
    # Calcola le differenze assolute
    au_real = np.array(au_real)
    au_avatar = np.array(au_avatar)

    # Controlla che au_real e au_avatar abbiano la stessa forma
    if au_real.shape != au_avatar.shape:
        raise ValueError("au_real e au_avatar devono avere la stessa forma.")

    # Controlla che il numero di colonne corrisponda alla lunghezza di au_names
    if au_real.shape[1] != len(au_names):
        raise ValueError("Il numero di colonne in au_real/au_avatar deve corrispondere alla lunghezza di au_names.")

    # Calcola le differenze assolute
    au_difference = np.abs(au_real - au_avatar)

    # Crea un DataFrame
    df_difference = pd.DataFrame(au_difference, columns=au_names)

    # Configura la dimensione della figura
    plt.figure(figsize=(12, 8))

    # Plotta la heatmap
    sns.heatmap(df_difference, cmap='viridis', cbar_kws={'label': 'Differenza AU'})

    # Aggiungi i titoli e le etichette
    plt.title('Heatmap delle Differenze tra le Action Units')
    plt.xlabel('Action Units')
    plt.ylabel('Indice dell\'immagine')
    plt.show()

def plot_au_scatter_plots(au_real, au_avatar, au_names):
    """
    Plotta scatter plot per ciascuna AU, mostrando la relazione tra i valori delle immagini e dell'avatar.

    Parametri:
    - au_real: array 2D delle AUs delle immagini reali.
    - au_avatar: array 2D delle AUs dell'avatar.
    - au_names: lista dei nomi delle Action Units.
    """
    import pandas as pd

    # Converte le liste in DataFrame
    df_real = pd.DataFrame(au_real, columns=au_names)
    df_avatar = pd.DataFrame(au_avatar, columns=au_names)

    # Numero di AUs
    num_aus = len(au_names)

    # Calcola il numero di righe e colonne per il subplot
    cols = 3
    rows = (num_aus + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))

    for idx, au in enumerate(au_names):
        plt.subplot(rows, cols, idx + 1)
        sns.scatterplot(x=df_real[au], y=df_avatar[au])
        plt.title(f'Scatter Plot {au}')
        plt.xlabel('Immagine')
        plt.ylabel('Avatar')
        # Calcola e mostra il coefficiente di Pearson
        coef, p_value = st.pearsonr(df_real[au], df_avatar[au])
        plt.text(0.05, 0.95, f'r = {coef:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()

def plot_pearson_correlation_boxplot(pearson_correlations):
    """
    Plotta un boxplot dei Coefficienti di Correlazione di Pearson.

    Parametri:
    - pearson_correlations: lista o array numpy di coefficienti di Pearson.
    """
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=pearson_correlations, color='lightgreen')
    plt.title('Boxplot dei Coefficienti di Pearson')
    plt.ylabel('Coefficiente di Pearson')
    plt.show()



run()