import json

from predict_endpoint import smiles_encode, predict

drug_1 = smiles_encode("CN1C2=C(C=C(C=C2)N(CCCl)CCCl)N=C1CCCC(=O)O")
drug_2 = smiles_encode("C1=CC=C2C(=C1)C(=NN2CC3=C(C=C(C=C3)Cl)Cl)C(=O)O")

features = json.loads(open("./drugcombdb/context_set_m.json", 'r').read())
cell_line = features["NCIH520"]

predicted = predict(drug_1["drug_smiles"], drug_2["drug_smiles"], cell_line, drug_1["drug_fp"], drug_2["drug_fp"])

print(predicted)
