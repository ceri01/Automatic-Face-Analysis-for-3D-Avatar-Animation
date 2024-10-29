# Automatic-Face-Analysis-for-3D-Avatar-Animation

## Setup
1. installa python3.11 (non è possibile usare una versione superiore di python perchè la libreria py-feat al 
suo interno utilizza una libreira chiamata nltools che non è aggiornata, e ad oggi (31/07/2024) non supporta python 12)
2. crea virtualenv usando python3.11 -m venv venv
3. aggiorna pip se non all'ultima versione (teoricamente dovrebbe essere meglio)
4. entra nel virtualenv
5. Per installare la freenect2 è necessario installare sul proprio sistema (libfreenect2)[https://github.com/OpenKinect/libfreenect2/blob/master/README.md#installation],
questo è necessario in caso si voglia usare il prototipo per acquisire lo stream da una kinect, ingorare se non serve, ovviamente però l'installazione tramite pip
di freenect2 fallirà.
6. Per poter usare lo screep di testing è necessario utilizzare un tool di face aimation, in questo caso viene usato (openFACS)[https://github.com/phuselab/openFACS], ed è 
possibile scaricare la demo eseguibile da (qui)[https://github.com/phuselab/openFACS/releases/download/1.0.1/openFACS_Linux.tar.gz]. È
importante scaricare all'interno di questa cartella questo eseguibile e anche la repo di openFACS, in modo che siano disponibili le funzioi python per comunicare con l'avatar.
7. Installa i requirements, usa le versioni presenti nel file requirements.txt, versioni più recenti non sono 
compatibili con librerie che non vengono aggiornate da un po.

