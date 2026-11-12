import math
import re
from pathlib import Path

import cmudict
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

CORPUS = Path("corpus")

PHONEMES = "aa ae ah ao aw ay b ch d dh eh er ey f g hh ih iy jh k l m n ng ow oy p r s sh t th uh uw v w y z zh".split()
SYMBOLES = ["<blanc>", *PHONEMES, "<debut>", "<fin>"]
BLANC, DEBUT, FIN = 0, len(SYMBOLES) - 2, len(SYMBOLES) - 1     # le blanc de la CTC sert aussi de remplissage

FREQUENCE = 16000
TAILLE_FFT, FENETRE, PAS = 512, 400, 160     # trames de 25 ms, une toutes les 10 ms
BANDES = 80


def construire_banc_mel():
    frequences = torch.linspace(0, FREQUENCE / 2, TAILLE_FFT // 2 + 1)[:, None]
    mel_max = 2595 * math.log10(1 + FREQUENCE / 2 / 700)
    sommets = 700 * (10 ** (torch.linspace(0, mel_max, BANDES + 2) / 2595) - 1)
    montees = (frequences - sommets[:-2]) / (sommets[1:-1] - sommets[:-2])
    descentes = (sommets[2:] - frequences) / (sommets[2:] - sommets[1:-1])
    return torch.minimum(montees, descentes).clamp(min=0)


BANC_MEL = construire_banc_mel()


def calculer_logmel(chemin_wav):
    signal, _ = sf.read(chemin_wav, dtype="float32")
    spectre = torch.stft(torch.from_numpy(signal), TAILLE_FFT, PAS, FENETRE, torch.hann_window(FENETRE),
                         center=False, return_complex=True)
    logmel = torch.log(spectre.abs().pow(2).T @ BANC_MEL + 1e-10)
    return (logmel - logmel.mean(dim=0)) / logmel.std(dim=0)


def lire_corpus():
    prononciations = cmudict.dict()
    corpus = {"apprentissage": [], "validation": [], "test": []}

    for ligne in (CORPUS / "etc" / "txt.done.data").read_text().splitlines():
        phrase, texte = ligne.strip("() ").split(maxsplit=1)
        mots = [mot.strip("'") for mot in re.findall(r"[a-z0-9']+", texte.lower())]
        if not all(mot in prononciations for mot in mots):
            continue    # un mot absent du dictionnaire : phrase écartée

        phonemes = [phoneme.rstrip("012").lower() for mot in mots for phoneme in prononciations[mot][0]]
        ensemble = {"0": "validation", "5": "test"}.get(phrase[-1], "apprentissage")
        corpus[ensemble].append({
            "texte": texte.strip('"'),
            "phonemes": [SYMBOLES.index(phoneme) for phoneme in phonemes],
            "logmel": calculer_logmel(CORPUS / "wav" / f"{phrase}.wav"),
        })

    return corpus


def assembler_lot(exemples):
    logmels = [exemple["logmel"] for exemple in exemples]
    longueurs = torch.tensor([len(logmel) for logmel in logmels])
    entrees = [torch.tensor([DEBUT, *exemple["phonemes"]]) for exemple in exemples]
    cibles = [torch.tensor([*exemple["phonemes"], FIN]) for exemple in exemples]
    return (pad_sequence(logmels, batch_first=True), longueurs,
            pad_sequence(entrees, batch_first=True), pad_sequence(cibles, batch_first=True))
