import numpy as np
from feat import Detector
from matplotlib import pyplot as plt
from TestUtils import catch_aus_libreface, catch_aus_feat

detector = Detector(face_model="retinaface", landmark_model="mobilefacenet", au_model='xgb')

au_labels_openFACS = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU25", "AU26", "AU28", "AU43"
]

ifeat = catch_aus_feat("", detector)
ilib = catch_aus_libreface("")

afeat = catch_aus_feat("", detector)
alib = catch_aus_libreface("")


def au_comparison_bar(img_aus, avatar_aus, au_labels, model_name):

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

au_comparison_bar(ifeat, afeat, au_labels_openFACS, "py-feat")
au_comparison_bar(ilib, alib, au_labels_openFACS, "libreface")