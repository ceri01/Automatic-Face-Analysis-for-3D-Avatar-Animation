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

dataset = "dataset/"
avatar_image_location = "avatar-screen/"
subprocess.Popen(["LinuxNoEditor/ActionUnitsFace.sh"])
time.sleep(15)

au_labels_openFACS = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU25", "AU26", "AU28", "AU43"
]

img_aus_feat_list = []
img_aus_libreface_list = []
avatar_aus_feat_list = []
avatar_aus_libreface_list = []

results_feat_cosim = []
results_libreface_consim = []
results_between_models_cosim = []

results_feat_pearson = []
results_libreface_pearson = []
results_between_models_pearson = []

times_libreface = []
times_feat = []

def run():
    avatar_image_number = 0
    detector = Detector(face_model="retinaface", landmark_model="mobilefacenet", au_model='xgb')

    for filename in os.listdir(dataset):
        avatar_aus_libreface = []
        avatar_aus_feat = []

        start = time.time()
        img_aus_libreface = catch_aus_libreface(dataset + filename)
        end = time.time()

        if img_aus_libreface:
            img_aus_libreface_list.append(img_aus_libreface)
            times_libreface.append(end - start)
            sendAUS(img_aus_libreface, 1)
            time.sleep(0.5)
            perform(avatar_image_location + "a" + str(avatar_image_number) + ".png")  # perform screenshot
            avatar_aus_libreface = catch_aus_libreface(avatar_image_location + str(avatar_image_number) + ".png")

        start = time.time()
        img_aus_feet = catch_aus_feat(dataset + filename, detector)
        end = time.time()

        if img_aus_feet:
            img_aus_feat_list.append(img_aus_feet)
            times_feat.append(end - start)
            sendAUS(img_aus_feet, 1)
            time.sleep(0.5)
            perform(avatar_image_location + "b" + str(avatar_image_number) + ".png")  # perform screenshot
            avatar_aus_feat = catch_aus_feat(avatar_image_location + str(avatar_image_number) + ".png", detector)

        avatar_image_number += 1

        if avatar_aus_libreface is None or avatar_aus_feat is None:
            continue

        avatar_aus_libreface_list.append(avatar_aus_libreface)
        avatar_aus_feat_list.append(avatar_aus_feat)

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

        print(avatar_image_number)

    print(f"pf aus img: {len(img_aus_feat_list)}\nlib aus img: {len(img_aus_libreface_list)}\npf aus avatar: {len(avatar_aus_feat_list)}\n lib aus avatar: {len(avatar_aus_libreface_list)}\n\n")

    print("\nresults of comparison between AUs caught from images and avatar using pyfeat and cosine similarity")
    make_results(results_feat_cosim)
    print("\nresults of comparison between AUs caught from images and avatar using pyfeat and pearson index")
    make_results(results_feat_pearson)


    print("\nresults of comparison between AUs caught from images and avatar using libreface and cosine similarity")
    make_results(results_libreface_consim)
    print("\nresults of comparison between AUs caught from images and avatar using libreface and pearson index")
    make_results(results_libreface_pearson)

    print("\nresults of comparison between AUs models using cosine similarity")
    make_results(results_between_models_cosim)
    print("\n\nresults of comparison between AUs models using pearson index")
    make_results(results_between_models_pearson)
    print("\n")

    feat_time_mean = np.mean(times_feat)
    libreface_time_mean = np.mean(times_libreface)
    print(f"\nmedia tempo py-feat {feat_time_mean}")
    print(f"media tempo libreface {libreface_time_mean}")

    # print grafici

    index_boxplot(results_feat_pearson, "Coefficienti di Pearson", "py-feat", 0, 1, 0.2)
    index_boxplot(results_libreface_pearson, "Coefficienti di Pearson", "libreface", 0, 1, 0.2)

    index_boxplot(results_libreface_consim, "Cosine similarity", "py-feat", 0, 1, 0.1)
    index_boxplot(results_feat_cosim, "Cosine similarity", "libreface", 0, 1, 0.1)


def make_results(final_results, confidence=0.95):
    dataset_size = len(final_results)
    aus_mean = np.mean(final_results)
    aus_dev_std = np.std(final_results)
    print(f"\tmedia AUs {aus_mean}")
    print(f"\tdev std AUs {aus_dev_std}")

    std_error = st.sem(final_results)  # Errore standard della media
    h = std_error * st.t.ppf((1 + confidence) / 2., dataset_size - 1)  # Calcola l'ampiezza dell'intervallo di confidenza

    lower_bound = aus_mean - h
    upper_bound = aus_mean + h
    print(f"\tIntervallo di confidenza ({confidence * 100}%): ({lower_bound}, {upper_bound})")
    # l'inetervallo di confidenza è un'intervallo (in questo caso 95%) tale per cui posso essere sicuro al 95% che il valore
    # reale (sconosciuto perchè noi stiamo facendo delle stime) della media (in questo caso della media, perchè stiamo considerando quella) sia
    # in quell'intervallo. In pratica ci indica quanto sono precise le stime, più è stretto migliori sono le stime

def au_scatter_plots(au_real, au_avatar, au_names, model):
    """
    Plotta scatter plot per ciascuna AU, mostrando la relazione tra i valori delle immagini e dell'avatar.

    Parametri:
    - au_real: array 2D delle AUs delle immagini reali.
    - au_avatar: array 2D delle AUs dell'avatar.
    - au_names: lista dei nomi delle Action Units.
    """

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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Scatter plot tra action units delle immagini e dell'avatar, ricavate con " +  model)
    plt.show()

def index_boxplot(index, label, model, x_min, x_max, x_interval):
    """
    Plotta un boxplot dei Coefficienti di Correlazione di Pearson.

    Parametri:
    - pearson_correlations: lista o array numpy di coefficienti di Pearson.
    """
    plt.figure(figsize=(10, 4))  # Ridotto l'altezza del grafico
    sns.boxplot(x=index, color='lightgreen')  # Boxplot orizzontale
    sns.stripplot(x=index, size=4, color=".3")  # Pallini
    plt.title('Boxplot dei ' + label + " usando " + model)
    plt.xlabel(label)
    plt.xticks(ticks=np.arange(x_min, x_max, x_interval))
    plt.show()


    # Mostra il grafico
    plt.show()

def au_comparison_bar(img_aus, avatar_aus, au_labels, model_name):
    """
    Funzione che crea un grafico a barre per confrontare le AUs tra immagine e avatar.

    Parametri:
    - img_aus: Lista di valori delle AUs estratte dall'immagine.
    - avatar_aus: Lista di valori delle AUs estratte dall'avatar.
    - au_labels: Lista di nomi delle AUs (etichette).
    - model_name: Nome del modello utilizzato per la visualizzazione nel titolo.
    """
    # Verifica che la lunghezza delle liste sia uguale
    assert len(img_aus) == len(avatar_aus) == len(au_labels), "Le lunghezze delle liste devono essere uguali."

    # Imposta la larghezza delle barre e la posizione sull'asse x
    x = np.arange(len(au_labels))  # Posizioni sull'asse x
    width = 0.35  # Larghezza delle barre

    # Creazione del grafico
    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(12)

    # Aggiungi le barre per le AUs delle immagini
    bars_img = ax.bar(x - width / 2, img_aus, align='edge', width=width, label='Immagine', color='skyblue')

    # Aggiungi le barre per le AUs dell'avatar
    bars_avatar = ax.bar(x + width / 2, avatar_aus, align='edge', width=width, label='Avatar', color='salmon')

    # Aggiungi etichette e titolo
    ax.set_xlabel('Action Units (AUs)')
    ax.set_ylabel('Attivazione AU')
    ax.set_title(f'Confronto delle AUs tra immagine e avatar usando {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(au_labels)
    ax.legend()

    # Mostra il grafico
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)
    plt.show()

run()