# readme

Un Transformer encodeur-décodeur écrit à la main en PyTorch : il écoute une phrase en anglais et écrit ses phonèmes.

mots clés :
encodeur (audio) + décodeur (phonèmes)
auto-attention + attention croisée
norme après le résidu (attention is all you need)
positionnal encoding sinusoïdales
CTC + entropie croisée
phrases de longueurs variables, remplissage

```bash
pip install -r requirements.txt
python entrainer.py
python demo.py
```

**44 % d'erreur** sur les phonèmes de 111 phrases jamais entendues, après 50 minutes d'audio.

![Attention croisée](attention.png)
