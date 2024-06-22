import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sentence_transformers import SentenceTransformer

from model_h import MultiViewNet

sentence_transformer = SentenceTransformer('output/simcsesqrt-model', device=torch.device("cuda"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiViewNet()
checkpoint = torch.load('mainsplit-attention-comb', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


def get_fingerprint(mol):
    return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)


def smiles_encode(smile_str):
    smile_encoded = sentence_transformer.encode(smile_str)
    mol = Chem.MolFromSmiles(smile_str)
    fp = np.asarray(get_fingerprint(mol))
    return {"drug_smiles": smile_encoded, "drug_fp": fp}


def predict(smile_1_vectors, smile_2_vectors, context, fp1_vectors, fp2_vectors):
    smile_1_vectors = torch.FloatTensor([smile_1_vectors]).to(device)
    smile_2_vectors = torch.FloatTensor([smile_2_vectors]).to(device)
    context = torch.FloatTensor([[context]]).to(device)
    fp1_vectors = torch.FloatTensor([fp1_vectors]).to(device)
    fp2_vectors = torch.FloatTensor([fp2_vectors]).to(device)

    return model.forward(smile_1_vectors, smile_2_vectors, context, fp1_vectors, fp2_vectors)
