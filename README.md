# IA_oso_player

Projet réalisé par les étudiants de l'ENSC [ROCHE Eléa](https://github.com/elroche) et [PERRET Quentin](https://github.com/QuentinPerret).

Un rapport détaillant le choix du projet, sa gestion, ainsi qu' une explication du code proposé est disponible au [lien suivant](https://github.com/elroche/IA_oso_player/blob/main/Rapport_IA_Perret_Roche.pdf).

## Installations
### Installations préliminaires
Afin d'utiliser ce projet plusieurs outils doivent être installé au préalable: 
-  tesseractOCR doit être installé, les guides d'installations pour toutes les OS sont disponible à ce [lien](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- poetry, un gestionnaire de dépendant doit aussi être installé et utilisable, les guides d'installations pour toutes les OS sont disponible à ce [lien](https://python-poetry.org/docs/#installation)

### Installtion du projet
Pour installer le projet et ses dépendances python utiliser les commandes suivantes : 
```console
foo@bar:~$ git clone https://github.com/elroche/IA_oso_player
foo@bar:~$ cd IA_oso_player
foo@IA_oso_player:~$ poetry shell
(ia-osu-player-cpu-py3.11) foo@IA_oso_player:~$ poetry install
```
Si vous souhaitez utiliser les versions cuda plutôt que les versions cpu de torch, un template [pyproject.cuda](https://github.com/elroche/IA_oso_player/blob/main/pyproject.cuda.toml) est disponible, de même le template [pyproject.cpu](https://github.com/elroche/IA_oso_player/blob/main/pyproject.cpu.toml) est disponible

## Utilisation
Les différents thread utilisables sont disponibles à l'aide de la commande : 
```console
(ia-osu-player-cpu-py3.11) foo@IA_oso_player:~$ python scripts/model_thread.py -h
usage: model_thread.py [-h] [-t] [-i] [-p]

All usable thread are presented here.

options:
  -h, --help       show this help message and exit
  -t, --tesseract  Lauch the tesseract thread.
  -i, --inference  Lauch the reinforcment model thread.
  -p, --pipeline   Lauch the pipeline thread.

```

Il est possible de lancer un thread en particulier de la manière suivante : 
```console
(ia-osu-player-cpu-py3.11) foo@IA_oso_player:~$ python scripts/model_thread.py -t
```
ou
```console
(ia-osu-player-cpu-py3.11) foo@IA_oso_player:~$ python scripts/model_thread.py --tesseract
```