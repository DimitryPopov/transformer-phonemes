import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from donnees import BLANC, SYMBOLES, assembler_lot, lire_corpus
from modele import Transformer

EPOQUES = 100
TAILLE_LOT = 16
TAUX_MAX = 5e-4
POIDS_CTC = 0.3

appareil = "cuda" if torch.cuda.is_available() else "cpu"


def calculer_perte(modele, lot):
    logmels, longueurs, entrees, cibles = (tenseur.to(appareil) for tenseur in lot)
    memoire, masque_audio = modele.encoder(logmels, longueurs)
    logits = modele.decoder(entrees, memoire, masque_audio)

    perte_decodeur = F.cross_entropy(logits.transpose(1, 2), cibles, ignore_index=BLANC)
    perte_ctc = F.ctc_loss(modele.tete_ctc(memoire).log_softmax(dim=-1).transpose(0, 1), cibles,
                           masque_audio.sum(dim=-1).flatten(), (cibles != BLANC).sum(dim=1) - 1)
    return (1 - POIDS_CTC) * perte_decodeur + POIDS_CTC * perte_ctc


torch.manual_seed(0)
corpus = lire_corpus()
apprentissage = DataLoader(corpus["apprentissage"], TAILLE_LOT, shuffle=True, collate_fn=assembler_lot)
validation = DataLoader(corpus["validation"], TAILLE_LOT, collate_fn=assembler_lot)

modele = Transformer(len(SYMBOLES)).to(appareil)
optimiseur = torch.optim.AdamW(modele.parameters())
planificateur = torch.optim.lr_scheduler.OneCycleLR(optimiseur, TAUX_MAX, total_steps=EPOQUES * len(apprentissage))
meilleure_perte = float("inf")

for epoque in range(1, EPOQUES + 1):
    modele.train()
    for lot in apprentissage:
        optimiseur.zero_grad()
        calculer_perte(modele, lot).backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        optimiseur.step()
        planificateur.step()

    modele.eval()
    with torch.no_grad():
        perte_validation = sum(calculer_perte(modele, lot).item() for lot in validation) / len(validation)
    print(f"époque {epoque}/{EPOQUES} : perte de validation {perte_validation:.3f}", flush=True)

    if perte_validation < meilleure_perte:
        meilleure_perte = perte_validation
        torch.save(modele.state_dict(), "modele.pt")
