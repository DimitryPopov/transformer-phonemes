import matplotlib.pyplot as plt
import torch

from donnees import DEBUT, FIN, SYMBOLES, lire_corpus
from modele import Transformer


@torch.no_grad()
def transcrire(modele, logmel):
    memoire, masque_audio = modele.encoder(logmel[None], torch.tensor([len(logmel)]))
    phonemes = [DEBUT]
    while phonemes[-1] != FIN and len(phonemes) < 150:
        # tout le décodeur est recalculé à chaque phonème : garder les clés et les valeurs en cache
        logits = modele.decoder(torch.tensor([phonemes]), memoire, masque_audio)
        phonemes.append(logits[0, -1].argmax().item())
    return phonemes[1:-1]


def distance_edition(reference, hypothese):
    distances = list(range(len(hypothese) + 1))
    for i, attendu in enumerate(reference, start=1):
        precedentes, distances = distances, [i]
        for j, obtenu in enumerate(hypothese, start=1):
            distances.append(min(precedentes[j] + 1, distances[j - 1] + 1, precedentes[j - 1] + (attendu != obtenu)))
    return distances[-1]


def en_symboles(indices):
    return " ".join(SYMBOLES[indice] for indice in indices)


modele = Transformer(len(SYMBOLES))
modele.load_state_dict(torch.load("modele.pt", map_location="cpu"))
modele.eval()

test = lire_corpus()["test"]
transcriptions = [transcrire(modele, exemple["logmel"]) for exemple in test]

erreurs = sum(distance_edition(exemple["phonemes"], transcription) for exemple, transcription in zip(test, transcriptions))
phonemes = sum(len(exemple["phonemes"]) for exemple in test)
print(f"taux d'erreur sur {len(test)} phrases jamais entendues : {erreurs / phonemes:.1%}")

for exemple, transcription in zip(test[:3], transcriptions):
    print()
    print(exemple["texte"])
    print("  attendu :", en_symboles(exemple["phonemes"]))
    print("  obtenu  :", en_symboles(transcription))

# la dernière transcription laisse ses poids d'attention dans le modèle
transcription = transcrire(modele, test[0]["logmel"])
poids = modele.decodeur[-1].attention_croisee.poids[0].mean(dim=0)

plt.figure(figsize=(10, 7))
plt.imshow(poids, origin="lower", aspect="auto")
plt.yticks(range(len(poids)), [SYMBOLES[indice] for indice in transcription] + ["<fin>"])
plt.xlabel("position dans l'audio (40 ms)")
plt.title(test[0]["texte"])
plt.savefig("attention.png", dpi=150, bbox_inches="tight")
