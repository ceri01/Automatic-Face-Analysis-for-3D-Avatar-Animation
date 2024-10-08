import subprocess
import pyautogui

def perform(filename):
    # Titolo della finestra da cercare
    window_title = "ActionUnitsFace"

    # Usa xdotool per ottenere l'ID della finestra
    result = subprocess.run(["xdotool", "search", "--name", window_title], stdout=subprocess.PIPE)
    window_id = result.stdout.decode().strip()

    if window_id:
        # Usa xdotool per ottenere la posizione e dimensione della finestra
        result = subprocess.run(["xdotool", "getwindowgeometry", "--shell", window_id], stdout=subprocess.PIPE)
        geometry = result.stdout.decode().splitlines()

        # Estrai le coordinate e le dimensioni
        x = int([line for line in geometry if line.startswith("X=")][0].split("=")[1])
        y = int([line for line in geometry if line.startswith("Y=")][0].split("=")[1])
        width = int([line for line in geometry if line.startswith("WIDTH=")][0].split("=")[1])
        height = int([line for line in geometry if line.startswith("HEIGHT=")][0].split("=")[1])

        # Fai uno screenshot della regione specificata
        screenshot = pyautogui.screenshot(region=(x, y, width, height))

        # Salva l'immagine
        screenshot.save(filename)
    else:
        print("Finestra non trovata")
