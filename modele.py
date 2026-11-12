import math

import torch
from torch import nn

DROPOUT = 0.1


def attention(requetes, cles, valeurs, masque):
    scores = requetes @ cles.transpose(-2, -1) / math.sqrt(requetes.shape[-1])
    poids = scores.masked_fill(~masque, float("-inf")).softmax(dim=-1)
    return poids @ valeurs, poids


def encodage_positionnel(x):
    longueur, dimension = x.shape[1:]
    positions = torch.arange(longueur, device=x.device)[:, None]
    frequences = 10000 ** (-torch.arange(0, dimension, 2, device=x.device) / dimension)
    angles = positions * frequences
    return torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(1)


def reseau_dense(dimension):
    return nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.ReLU(), nn.Linear(4 * dimension, dimension))


class AttentionMultiTetes(nn.Module):
    def __init__(self, dimension, tetes):
        super().__init__()
        self.tetes = tetes
        self.vers_requetes = nn.Linear(dimension, dimension)
        self.vers_cles = nn.Linear(dimension, dimension)
        self.vers_valeurs = nn.Linear(dimension, dimension)
        self.vers_sortie = nn.Linear(dimension, dimension)

    def separer_tetes(self, x):
        return x.unflatten(-1, (self.tetes, -1)).transpose(1, 2)

    def forward(self, x, source, masque):
        requetes = self.separer_tetes(self.vers_requetes(x))
        cles = self.separer_tetes(self.vers_cles(source))
        valeurs = self.separer_tetes(self.vers_valeurs(source))
        sortie, self.poids = attention(requetes, cles, valeurs, masque)
        return self.vers_sortie(sortie.transpose(1, 2).flatten(-2))


class BlocEncodeur(nn.Module):
    def __init__(self, dimension, tetes):
        super().__init__()
        self.auto_attention = AttentionMultiTetes(dimension, tetes)
        self.reseau_dense = reseau_dense(dimension)
        self.norme_attention = nn.LayerNorm(dimension)
        self.norme_dense = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, audio, masque_audio):
        audio = self.norme_attention(audio + self.dropout(self.auto_attention(audio, audio, masque_audio)))
        return self.norme_dense(audio + self.dropout(self.reseau_dense(audio)))


class BlocDecodeur(nn.Module):
    def __init__(self, dimension, tetes):
        super().__init__()
        self.auto_attention = AttentionMultiTetes(dimension, tetes)
        self.attention_croisee = AttentionMultiTetes(dimension, tetes)
        self.reseau_dense = reseau_dense(dimension)
        self.norme_auto = nn.LayerNorm(dimension)
        self.norme_croisee = nn.LayerNorm(dimension)
        self.norme_dense = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, phonemes, masque_causal, memoire, masque_audio):
        phonemes = self.norme_auto(phonemes + self.dropout(self.auto_attention(phonemes, phonemes, masque_causal)))
        phonemes = self.norme_croisee(phonemes + self.dropout(self.attention_croisee(phonemes, memoire, masque_audio)))
        return self.norme_dense(phonemes + self.dropout(self.reseau_dense(phonemes)))


class Transformer(nn.Module):
    def __init__(self, symboles, bandes=80, dimension=256, tetes=4, blocs_encodeur=4, blocs_decodeur=2):
        super().__init__()
        self.convolutions = nn.Sequential(nn.Conv1d(bandes, dimension, 3, stride=2, padding=1), nn.ReLU(),
                                          nn.Conv1d(dimension, dimension, 3, stride=2, padding=1), nn.ReLU())
        self.encodeur = nn.ModuleList(BlocEncodeur(dimension, tetes) for _ in range(blocs_encodeur))
        self.plongement = nn.Embedding(symboles, dimension)
        self.decodeur = nn.ModuleList(BlocDecodeur(dimension, tetes) for _ in range(blocs_decodeur))
        self.tete_decodeur = nn.Linear(dimension, symboles)
        self.tete_ctc = nn.Linear(dimension, symboles)
        self.dropout = nn.Dropout(DROPOUT)

    def encoder(self, logmels, longueurs):
        audio = self.convolutions(logmels.transpose(1, 2)).transpose(1, 2)      # 4 fois moins de positions
        positions = torch.arange(audio.shape[1], device=audio.device)
        masque_audio = (positions < (longueurs[:, None] + 3) // 4)[:, None, None, :]
        audio = self.dropout(audio + encodage_positionnel(audio))
        for bloc in self.encodeur:
            audio = bloc(audio, masque_audio)
        return audio, masque_audio

    def decoder(self, entrees, memoire, masque_audio):
        longueur = entrees.shape[1]
        masque_causal = torch.ones(longueur, longueur, dtype=torch.bool, device=entrees.device).tril()
        phonemes = self.plongement(entrees)
        phonemes = self.dropout(phonemes + encodage_positionnel(phonemes))
        for bloc in self.decodeur:
            phonemes = bloc(phonemes, masque_causal, memoire, masque_audio)
        return self.tete_decodeur(phonemes)
