'''
Stacking CNN + Transformer + Graph Neural Network for Active Learning in Molecular Property Prediction  
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, precision_recall_curve, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from tqdm import tqdm
import umap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# === 0.2. Dynamic Batch Size Helper ===
def get_dynamic_batch_size(dataset_size, max_batch_size=32):
    """
    Dynamically adjusts batch size for small datasets to avoid BatchNorm errors.
    """
    if dataset_size < 4:
        return 2  # Minimum batch size to prevent BatchNorm error
    
    # Find the largest power of 2 less than or equal to the dataset size, up to max_batch_size
    batch_size = 1
    while (batch_size * 2) <= min(dataset_size, max_batch_size):
        batch_size *= 2
        
    return batch_size


# === 0.5. Fingerprint and Descriptor Calculation Functions ===

def calculate_ecfp(df, smiles_col, radius=10, nBits=4096):
    '''
    Compute ECFP fingerprints, radius = 10, nBits = 4096
    ------
    df: DataFrame
    smiles_col: SMILE column
    '''
    def get_ecfp(smiles):
       try:
           mol = Chem.MolFromSmiles(smiles)
           if mol is None:
               print(f"SMILES conversion failed for: {smiles}")
               return [None] * nBits
           fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
           return [int(bit) for bit in fingerprint.ToBitString()]
       except Exception as e:
           print(f"Error processing SMILES {smiles}: {e}")
           return [None] * nBits  # Return a list of None if an error occurs
    ecfp_bits_df = df[smiles_col].apply(get_ecfp).apply(pd.Series)
    ecfp_bits_df.columns = [f'ECFP{i}' for i in range(nBits)]
    ecfp_bits_df
    return ecfp_bits_df

def calculate_rdkit(df, smiles_col, nBits=2048):
    '''
    Compute RDKIT fingerprints, nBits = 2048
    ------
    df: DataFrame
    smiles_col: SMILE column
    '''
    def get_rdkit(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            fingerprint = Chem.RDKFingerprint(mol)
            return [int(bit) for bit in fingerprint.ToBitString()]
        except:
            return [None] * nBits  # Return a list of None if an error occurs

    rdkit_bits_df = df[smiles_col].apply(get_rdkit).apply(pd.Series)
    rdkit_bits_df.columns = [f'RDKit{i}' for i in range(nBits)]
    return rdkit_bits_df

def calculate_maccs(df, smiles_col):
    '''
    Compute MACCS fingerprints, nBits = 167
    ------
    df: DataFrame
    smiles_col: SMILE column
    '''
    def get_maccs(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            return [int(bit) for bit in fingerprint.ToBitString()]
        except:
            return [None] * 167

    maccs_bits_df = df[smiles_col].apply(get_maccs).apply(pd.Series)
    maccs_bits_df.columns = [f'MACCS{i}' for i in range(167)]
    return maccs_bits_df

def calculate_descriptors(df, smiles_col):
    """
    Compute molecular descriptors using RDKit.
    ------
    df: DataFrame
    smiles_col: Column name containing SMILES strings
    """
    descriptor_functions = {
        'molecular_weight': Descriptors.MolWt,
        'log_p': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'CalcTPSA': rdMolDescriptors.CalcTPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'CalcNumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles,
        'CalcNumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
        'CalcNumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings,
        'CalcNumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms,
        'CalcNumRings': rdMolDescriptors.CalcNumRings,
        'CalcNumHeavyAtoms': rdMolDescriptors.CalcNumHeavyAtoms,
        'CalcNumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
        'CalcNumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles,
        'CalcNumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles,
        'NumValenceElectrons': Descriptors.NumValenceElectrons,
        'CalcNumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
        'CalcNumHeterocycles': rdMolDescriptors.CalcNumHeterocycles,
        'CalcNumAmideBonds': rdMolDescriptors.CalcNumAmideBonds,
    }

    def get_descriptors(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            return [func(mol) for func in descriptor_functions.values()]
        except:
            return [None] * len(descriptor_functions)

    descriptors_df = df[smiles_col].apply(get_descriptors).apply(pd.Series)
    descriptors_df.columns = list(descriptor_functions.keys())
    return descriptors_df


# === 1. SMILES Tokenizer and Vocabulary ===
SMILES_CHARS = [
        '<pad>', '<sos>', '<eos>', '<SEP>', '<MASK>', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '.', 
        'n', '3', 'F', 'Cl', '>>', '~', '-', '4', '[C@H]', 'S', '[C@@H]', '[O-]', 'Br', '#', '/', '[nH]', 
        '[N+]', 's', '5', 'o', 'P', '[Na+]', '[Si]', 'I', '[Na]', '[Pd]', '[K+]', '[K]', '[P]', 'B', '[C@]', 
        '[C@@]', '[Cl-]', '6', '[OH-]', '\\', '[N-]', '[Li]', '[H]', '[2H]', '[NH4+]', '[c-]', '[P-]', '[Cs+]',
        '[Li+]', '[Cs]', '[NaH]', '[H-]', '[O+]', '[BH4-]', '[Cu]', '7', '[Mg]', '[Fe+2]', '[n+]', '[Sn]', 
        '[BH-]', '[Pd+2]', '[CH]', '[I-]', '[Br-]', '[C-]', '[Zn]', '[B-]', '[F-]', '[Al]', '[P+]', '[BH3-]',
        '[Fe]', '[C]', '[AlH4]', '[Ni]', '[SiH]', '8', '[Cu+2]', '[Mn]', '[AlH]', '[nH+]', '[AlH4-]', '[O-2]',
        '[Cr]', '[Mg+2]', '[NH3+]', '[S@]', '[Pt]', '[Al+3]', '[S@@]', '[S-]', '[Ti]', '[Zn+2]', '[PH]', 
        '[NH2+]', '[Ru]', '[Ag+]', '[S+]', '[I+3]', '[NH+]', '[Ca+2]', '[Ag]', '9', '[Os]', '[Se]', '[SiH2]',
        '[Ca]', '[Ti+4]', '[Ac]', '[Cu+]', '[S]', '[Rh]', '[Cl+3]', '[cH-]', '[Zn+]', '[O]', '[Cl+]', '[SH]', 
        '[H+]', '[Pd+]', '[se]', '[PH+]', '[I]', '[Pt+2]', '[C+]', '[Mg+]', '[Hg]', '[W]', '[SnH]', '[SiH3]',
        '[Fe+3]', '[NH]', '[Mo]', '[CH2+]', '%10', '[CH2-]', '[CH2]', '[n-]', '[Ce+4]', '[NH-]', '[Co]', 
        '[I+]', '[PH2]', '[Pt+4]', '[Ce]', '[B]', '[Sn+2]', '[Ba+2]', '%11', '[Fe-3]', '[18F]', '[SH-]', 
        '[Pb+2]', '[Os-2]', '[Zr+4]', '[N]', '[Ir]', '[Bi]', '[Ni+2]', '[P@]', '[Co+2]', '[s+]', '[As]', 
        '[P+3]', '[Hg+2]', '[Yb+3]', '[CH-]', '[Zr+2]', '[Mn+2]', '[CH+]', '[In]', '[KH]', '[Ce+3]', '[Zr]',
        '[AlH2-]', '[OH2+]', '[Ti+3]', '[Rh+2]', '[Sb]', '[S-2]', '%12', '[P@@]', '[Si@H]', '[Mn+4]', 'p', 
        '[Ba]', '[NH2-]', '[Ge]', '[Pb+4]', '[Cr+3]', '[Au]', '[LiH]', '[Sc+3]', '[o+]', '[Rh-3]', '%13', 
        '[Br]', '[Sb-]', '[S@+]', '[I+2]', '[Ar]', '[V]', '[Cu-]', '[Al-]', '[Te]', '[13c]', '[13C]', '[Cl]', 
        '[PH4+]', '[SiH4]', '[te]', '[CH3-]', '[S@@+]', '[Rh+3]', '[SH+]', '[Bi+3]', '[Br+2]', '[La]', 
        '[La+3]', '[Pt-2]', '[N@@]', '[PH3+]', '[N@]', '[Si+4]', '[Sr+2]', '[Al+]', '[Pb]', '[SeH]', '[Si-]', 
        '[V+5]', '[Y+3]', '[Re]', '[Ru+]', '[Sm]', '*', '[3H]', '[NH2]', '[Ag-]', '[13CH3]', '[OH+]', '[Ru+3]',
        '[OH]', '[Gd+3]', '[13CH2]', '[In+3]', '[Si@@]', '[Si@]', '[Ti+2]', '[Sn+]', '[Cl+2]', '[AlH-]', 
        '[Pd-2]', '[SnH3]', '[B+3]', '[Cu-2]', '[Nd+3]', '[Pb+3]', '[13cH]', '[Fe-4]', '[Ga]', '[Sn+4]', 
        '[Hg+]', '[11CH3]', '[Hf]', '[Pr]', '[Y]', '[S+2]', '[Cd]', '[Cr+6]', '[Zr+3]', '[Rh+]', '[CH3]', 
        '[N-3]', '[Hf+2]', '[Th]', '[Sb+3]', '%14', '[Cr+2]', '[Ru+2]', '[Hf+4]', '[14C]', '[Ta]', '[Tl+]', 
        '[B+]', '[Os+4]', '[PdH2]', '[Pd-]', '[Cd+2]', '[Co+3]', '[S+4]', '[Nb+5]', '[123I]', '[c+]', '[Rb+]',
        '[V+2]', '[CH3+]', '[Ag+2]', '[cH+]', '[Mn+3]', '[Se-]', '[As-]', '[Eu+3]', '[SH2]', '[Sm+3]', '[IH+]',
        '%15', '[OH3+]', '[PH3]', '[IH2+]', '[SH2+]', '[Ir+3]', '[AlH3]', '[Sc]', '[Yb]', '[15NH2]', '[Lu]', 
        '[sH+]', '[Gd]', '[18F-]', '[SH3+]', '[SnH4]', '[TeH]', '[Si@@H]', '[Ga+3]', '[CaH2]', '[Tl]', 
        '[Ta+5]', '[GeH]', '[Br+]', '[Sr]', '[Tl+3]', '[Sm+2]', '[PH5]', '%16', '[N@@+]', '[Au+3]', '[C-4]',
        '[Nd]', '[Ti+]', '[IH]', '[N@+]', '[125I]', '[Eu]', '[Sn+3]', '[Nb]', '[Er+3]', '[123I-]', '[14c]',
        '%17', '[SnH2]', '[YH]', '[Sb+5]', '[Pr+3]', '[Ir+]', '[N+3]', '[AlH2]', '[19F]', '%18', '[Tb]', 
        '[14CH]', '[Mo+4]', '[Si+]', '[BH]', '[Be]', '[Rb]', '[pH]', '%19', '%20', '[Xe]', '[Ir-]', '[Be+2]', 
        '[C+4]', '[RuH2]', '[15NH]', '[U+2]', '[Au-]', '%21', '%22', '[Au+]', '[15n]', '[Al+2]', '[Tb+3]', 
        '[15N]', '[V+3]', '[W+6]', '[14CH3]', '[Cr+4]', '[ClH+]', 'b', '[Ti+6]', '[Nd+]', '[Zr+]', '[PH2+]', 
        '[Fm]', '[N@H+]', '[RuH]', '[Dy+3]', '%23', '[Hf+3]', '[W+4]', '[11C]', '[13CH]', '[Er]', '[124I]', 
        '[LaH]', '[F]', '[siH]', '[Ga+]', '[Cm]', '[GeH3]', '[IH-]', '[U+6]', '[SeH+]', '[32P]', '[SeH-]',
        '[Pt-]', '[Ir+2]', '[se+]', '[U]', '[F+]', '[BH2]', '[As+]', '[Cf]', '[ClH2+]', '[Ni+]', '[TeH3]',
        '[SbH2]', '[Ag+3]', '%24', '[18O]', '[PH4]', '[Os+2]', '[Na-]', '[Sb+2]', '[V+4]', '[Ho+3]', '[68Ga]',
        '[PH-]', '[Bi+2]', '[Ce+2]', '[Pd+3]', '[99Tc]', '[13C@@H]', '[Fe+6]', '[c]', '[GeH2]', '[10B]',
        '[Cu+3]', '[Mo+2]', '[Cr+]', '[Pd+4]', '[Dy]', '[AsH]', '[Ba+]', '[SeH2]', '[In+]', '[TeH2]', '[BrH+]',
        '[14cH]', '[W+]', '[13C@H]', '[AsH2]', '[In+2]', '[N+2]', '[N@@H+]', '[SbH]', '[60Co]', '[AsH4+]',
        '[AsH3]', '[18OH]', '[Ru-2]', '[Na-2]', '[CuH2]', '[31P]', '[Ti+5]', '[35S]', '[P@@H]', '[ArH]', 
        '[Co+]', '[Zr-2]', '[BH2-]', '[131I]', '[SH5]', '[VH]', '[B+2]', '[Yb+2]', '[14C@H]', '[211At]', 
        '[NH3+2]', '[IrH]', '[IrH2]', '[Rh-]', '[Cr-]', '[Sb+]', '[Ni+3]', '[TaH3]', '[Tl+2]', '[64Cu]',
        '[Tc]', '[Cd+]', '[1H]', '[15nH]', '[AlH2+]', '[FH+2]', '[BiH3]', '[Ru-]', '[Mo+6]', '[AsH+]',
        '[BaH2]', '[BaH]', '[Fe+4]', '[229Th]', '[Th+4]', '[As+3]', '[NH+3]', '[P@H]', '[Li-]', '[7NaH]',
        '[Bi+]', '[PtH+2]', '[p-]', '[Re+5]', '[NiH]', '[Ni-]', '[Xe+]', '[Ca+]', '[11c]', '[Rh+4]', '[AcH]',
        '[HeH]', '[Sc+2]', '[Mn+]', '[UH]', '[14CH2]', '[SiH4+]', '[18OH2]', '[Ac-]', '[Re+4]', '[118Sn]',
        '[153Sm]', '[P+2]', '[9CH]', '[9CH3]', '[Y-]', '[NiH2]', '[Si+2]', '[Mn+6]', '[ZrH2]', '[C-2]',
        '[Bi+5]', '[24NaH]', '[Fr]', '[15CH]', '[Se+]', '[At]', '[P-3]', '[124I-]', '[CuH2-]', '[Nb+4]',
        '[Nb+3]', '[MgH]', '[Ir+4]', '[67Ga+3]', '[67Ga]', '[13N]', '[15OH2]', '[2NH]', '[Ho]', '[Cn]'
    ]
SMILES_VOCAB_SIZE = len(SMILES_CHARS)
char_to_idx = {char: i for i, char in enumerate(SMILES_CHARS)}
idx_to_char = {i: char for i, char in enumerate(SMILES_CHARS)}
MAX_SMILES_LEN = 200  # Max length for padding

def tokenize_smiles(smiles):
    """Tokenizes a SMILES string."""
    tokens = ['<sos>'] + list(smiles) + ['<eos>']
    return [char_to_idx.get(char, char_to_idx['.']) for char in tokens]


# === 1. SMILES to Graph Conversion ===
def atom_features(atom):
    """Extracts features for a single atom."""
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        float(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        float(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)

def bond_features(bond):
    """Extracts features for a single bond."""
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        float(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)

def mol_to_graph(smiles, label=None):
    """Converts a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        feat = bond_features(bond)
        edge_attr.extend([feat, feat])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 4), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data


# === 1.5. Dataset Class ===
class MolecularDataset(Dataset):
    """Fingerprints, SMILES, and Label data set"""
    def __init__(self, desc, ecfp, maccs, rdkit, smiles, labels=None):
        self.desc   = torch.FloatTensor(desc.values if isinstance(desc, pd.DataFrame) else desc)
        self.ecfp   = torch.FloatTensor(ecfp.values if isinstance(ecfp, pd.DataFrame) else ecfp)
        self.maccs  = torch.FloatTensor(maccs.values if isinstance(maccs, pd.DataFrame) else maccs)
        self.rdkit  = torch.FloatTensor(rdkit.values if isinstance(rdkit, pd.DataFrame) else rdkit)
        self.smiles = smiles
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.desc)
    
    def __getitem__(self, idx):
        smiles_str = self.smiles[idx]
        tokens = tokenize_smiles(smiles_str)
        
        item = {
            'desc' : self.desc[idx],
            'ecfp' : self.ecfp[idx],
            'maccs': self.maccs[idx],
            'rdkit': self.rdkit[idx],
            'smiles': smiles_str,
            'smiles_tokens': torch.LongTensor(tokens)
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

def collate_fn(batch):
    """Custom collate function to handle graph and SMILES padding."""
    desc_list, ecfp_list, maccs_list, rdkit_list, label_list, graph_list, smiles_tokens_list, smiles_list = [], [], [], [], [], [], [], []
    
    for item in batch:
        graph = mol_to_graph(item['smiles'])
        if graph is None: continue # Skip invalid SMILES

        desc_list.append(item['desc'])
        ecfp_list.append(item['ecfp'])
        maccs_list.append(item['maccs'])
        rdkit_list.append(item['rdkit'])
        graph_list.append(graph)
        smiles_list.append(item['smiles'])
        
        # Truncate or pad smiles tokens
        tokens = item['smiles_tokens']
        if len(tokens) > MAX_SMILES_LEN:
            tokens = torch.cat([
                tokens[:MAX_SMILES_LEN-1],
                torch.LongTensor([char_to_idx['<eos>']])
            ])
        smiles_tokens_list.append(tokens)

        if 'label' in item:
            label_list.append(item['label'])

    if not graph_list: return None

    # Pad SMILES tokens
    padded_smiles = nn.utils.rnn.pad_sequence(smiles_tokens_list, batch_first=True, padding_value=char_to_idx['<pad>'])

    collated_batch = {
        'desc': torch.stack(desc_list),
        'ecfp': torch.stack(ecfp_list),
        'maccs': torch.stack(maccs_list),
        'rdkit': torch.stack(rdkit_list),
        'graph_data': Batch.from_data_list(graph_list),
        'smiles_tokens': padded_smiles,
        'smiles': smiles_list
    }
    if label_list:
        collated_batch['label'] = torch.stack(label_list)
        
    return collated_batch


# === 2. CNN Module ===
class CNN_Module(nn.Module):
    """CNN for feature extraction from fingerprints"""
    def __init__(self, input_dim, output_dim=128):
        super(CNN_Module, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, output_dim)
        return x

# === 3. Transformer Module ===
class TransformerModule(nn.Module):
    """Transformer for feature extraction"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, output_dim=64):
        super(TransformerModule, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.embedding(x)  # (batch, 1, d_model)
        x = self.transformer(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, output_dim)
        return x

# === 3.5 GNN Module ===
class GNN_Module(nn.Module):
    """Enhanced GNN with more layers and skip connections"""
    def __init__(self, input_dim=8, feature_dim=256):  # Increase capacity
        super(GNN_Module, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = GCNConv(256, 256)  # Add third layer
        self.bn3 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        
        # Layer 2
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)
        
        # Layer 3 with skip connection
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = x3 + x2  # Residual connection
        x3 = self.dropout(x3)
        
        # Global pooling
        x = global_mean_pool(x3, batch)
        x = self.fc(x)
        
        return x


# === 3.6 SMILES Decoder Module ===
class SmilesDecoder(nn.Module):
    """Transformer Decoder for SMILES reconstruction"""
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, latent_dim):
        super(SmilesDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_SMILES_LEN, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.latent_to_memory = nn.Linear(latent_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt_tokens, memory):
        # tgt_tokens shape: (batch, seq_len)
        # memory shape: (batch, latent_dim)
        
        tgt_embed = self.embedding(tgt_tokens) + self.pos_encoder[:, :tgt_tokens.size(1), :]
        
        # Project latent vector to match decoder dimension and repeat for each token
        memory_proj = self.latent_to_memory(memory).unsqueeze(1).repeat(1, tgt_tokens.size(1), 1)
        
        # Generate a mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
        
        output = self.transformer_decoder(tgt_embed, memory_proj, tgt_mask=tgt_mask)
        return self.fc_out(output)


# === 4. CytoNet Model ===
class Multimodal(nn.Module):
    """Enhanced Multimodal with better architecture"""
    def __init__(self, desc_dim, ecfp_dim, maccs_dim, rdkit_dim, feature_dim=256):  # Increase to 256
        super(Multimodal, self).__init__()
        
        # --- Encoder Part with Residual Connections ---
        self.cnn_desc  = CNN_Module(desc_dim, feature_dim)
        self.cnn_ecfp  = CNN_Module(ecfp_dim, feature_dim)
        self.cnn_maccs = CNN_Module(maccs_dim, feature_dim)
        self.cnn_rdkit = CNN_Module(rdkit_dim, feature_dim)
        
        # Transformer with more capacity
        self.trans_desc  = TransformerModule(desc_dim, d_model=256, nhead=8, num_layers=3, output_dim=feature_dim)
        self.trans_ecfp  = TransformerModule(ecfp_dim, d_model=256, nhead=8, num_layers=3, output_dim=feature_dim)
        self.trans_maccs = TransformerModule(maccs_dim, d_model=256, nhead=8, num_layers=3, output_dim=feature_dim)
        self.trans_rdkit = TransformerModule(rdkit_dim, d_model=256, nhead=8, num_layers=3, output_dim=feature_dim)

        # Enhanced GNN
        self.gnn = GNN_Module(input_dim=8, feature_dim=feature_dim)
        
        # 🔥 IMPROVED: Deeper fusion with residual connections and attention
        fusion_dim = feature_dim * 9
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 🔥 ADD: Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 9),  # 9 feature groups
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        # Decoder
        self.smiles_decoder = SmilesDecoder(
            vocab_size=SMILES_VOCAB_SIZE,
            embed_dim=128,
            nhead=4,
            num_layers=3,
            latent_dim=128
        )
        
    def forward(self, desc, ecfp, maccs, rdkit, graph_data, smiles_tokens):
        # --- Encoder ---
        cnn_desc_feat = self.cnn_desc(desc)
        cnn_ecfp_feat = self.cnn_ecfp(ecfp)
        cnn_maccs_feat = self.cnn_maccs(maccs)
        cnn_rdkit_feat = self.cnn_rdkit(rdkit)
        
        trans_desc_feat = self.trans_desc(desc)
        trans_ecfp_feat = self.trans_ecfp(ecfp)
        trans_maccs_feat = self.trans_maccs(maccs)
        trans_rdkit_feat = self.trans_rdkit(rdkit)

        gnn_feat = self.gnn(graph_data)
        
        # 🔥 Concatenate with attention weighting
        combined = torch.cat([
            cnn_desc_feat, cnn_ecfp_feat, cnn_maccs_feat, cnn_rdkit_feat,
            trans_desc_feat, trans_ecfp_feat, trans_maccs_feat, trans_rdkit_feat,
            gnn_feat
        ], dim=1)
        
        # 🔥 Apply feature attention
        attention_weights = self.feature_attention(combined).unsqueeze(2)  # (batch, 9, 1)
        feature_groups = combined.view(combined.size(0), 9, -1)  # (batch, 9, feature_dim)
        weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
        
        # Fusion
        latent_features = self.fusion(weighted_features)
        
        # Classification
        logits = self.classifier(latent_features)
        
        # Decoder
        decoder_input = smiles_tokens[:, :-1]
        reconstruction_logits = self.smiles_decoder(decoder_input, latent_features)
        
        return logits, latent_features, reconstruction_logits
    
    def predict_proba(self, desc, ecfp, maccs, rdkit, graph_data):
        """Get probability predictions. Ignores decoder for prediction."""
        with torch.no_grad():
            # Create dummy tokens for forward pass during prediction
            batch_size = desc.size(0)
            dummy_tokens = torch.zeros((batch_size, 2), dtype=torch.long).to(desc.device)
            logits, _, _ = self.forward(desc, ecfp, maccs, rdkit, graph_data, dummy_tokens)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()


# === 4.5. Focal Loss ===
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are the logits from the model (batch_size, C)
        # targets are the ground truth labels (batch_size)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# === 5. Training Function ===
def train_multimodal(model, train_loader, criterion_cls, criterion_recon, optimizer, device, recon_weight=0.2):
    """Train Multimodal for one epoch with combined loss"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        if batch is None: continue
        desc = batch['desc'].to(device)
        ecfp = batch['ecfp'].to(device)
        maccs = batch['maccs'].to(device)
        rdkit = batch['rdkit'].to(device)
        graph_data = batch['graph_data'].to(device)
        labels = batch['label'].to(device)
        smiles_tokens = batch['smiles_tokens'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        cls_logits, _, recon_logits = model(desc, ecfp, maccs, rdkit, graph_data, smiles_tokens)
        
        # --- Calculate Losses ---
        # 1. Classification Loss
        loss_cls = criterion_cls(cls_logits, labels)
        
        # 2. Reconstruction Loss
        # Target is tokens shifted by one, excluding the first one (<sos>)
        recon_target = smiles_tokens[:, 1:]
        # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize) and (Batch * SeqLen)
        loss_recon = criterion_recon(
            recon_logits.reshape(-1, SMILES_VOCAB_SIZE),
            recon_target.reshape(-1)
        )
        
        # Combined Loss
        loss = loss_cls + recon_weight * loss_recon
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = cls_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


# === 5.5. Train with Validation ===
def train_with_validation(model, full_train_dataset, epochs, criterion_cls, criterion_recon, optimizer, scheduler, device):
    """
    Trains a model, using a validation split to find the best model state.
    Returns the best model state dict and training history.
    """
    dataset_size = len(full_train_dataset)
    batch_size = get_dynamic_batch_size(dataset_size)

    # If the dataset is too small for a validation split, train on the whole thing
    if dataset_size < batch_size * 1.25: # Heuristic: need at least a full batch for train and some for val
        print(f"⚠️  Dataset too small for validation split (size={dataset_size}), training on full dataset with batch_size={batch_size}.")
        sub_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        sub_val_loader = None # No validation
    else:
        # Split the full training data set nto sub-train and sub-validation sets
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        sub_train_dataset, sub_val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
        
        # Adjust batch size for the smaller training split
        train_batch_size = get_dynamic_batch_size(len(sub_train_dataset))
        print(f"   - Using dynamic batch size: {train_batch_size} for training split.")

        sub_train_loader = DataLoader(sub_train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        sub_val_loader = DataLoader(sub_val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    best_val_auprc = 0.0
    best_model_state = model.state_dict() # Start with current state
    history = {'loss': [], 'acc': [], 'val_auprc': []}

    for epoch in range(epochs):
        loss, acc = train_multimodal(model, sub_train_loader, criterion_cls, criterion_recon, optimizer, device)
        history['loss'].append(loss)
        history['acc'].append(acc)

        # Evaluate on the sub-validation set if it exists
        if sub_val_loader:
            val_probs, val_labels = evaluate_multimodal(model, sub_val_loader, device)
            # Use probabilities of the positive class (class 1) for AUPRC
            val_auprc = average_precision_score(val_labels, val_probs[:, 1])
            history['val_auprc'].append(val_auprc)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Val AUPRC: {val_auprc:.4f}")

            scheduler.step(val_auprc)

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = model.state_dict()
        else:
            # If no validation, just save the last epoch's model
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | No validation.")
            best_model_state = model.state_dict()
            history['val_auprc'].append(0.0) # Append 0 as a placeholder
            
    return best_model_state, history


# === 6. Evaluation Function ===
def evaluate_multimodal(model, data_loader, device):
    """Evaluate Multimodal and return predictions"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            
            # Create dummy tokens for evaluation pass
            batch_size = desc.size(0)
            dummy_tokens = torch.zeros((batch_size, 2), dtype=torch.long).to(device)
            
            logits, _, _ = model(desc, ecfp, maccs, rdkit, graph_data, dummy_tokens)
            probs = F.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            if 'label' in batch:
                all_labels.append(batch['label'].cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    if all_labels:
        all_labels = np.concatenate(all_labels)
        return all_probs, all_labels
    return all_probs


# === 7. Active Learning Sampling Strategies ===

def get_latent_embeddings(model, dataset, device):
    """Helper function to get latent space embeddings for a dataset."""
    model.eval()
    embeddings = []
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            
            # Encoder pass
            cnn_desc_feat = model.cnn_desc(desc)
            cnn_ecfp_feat = model.cnn_ecfp(ecfp)
            cnn_maccs_feat = model.cnn_maccs(maccs)
            cnn_rdkit_feat = model.cnn_rdkit(rdkit)
            trans_desc_feat = model.trans_desc(desc)
            trans_ecfp_feat = model.trans_ecfp(ecfp)
            trans_maccs_feat = model.trans_maccs(maccs)
            trans_rdkit_feat = model.trans_rdkit(rdkit)
            gnn_feat = model.gnn(graph_data)
            
            combined = torch.cat([
                cnn_desc_feat, cnn_ecfp_feat, cnn_maccs_feat, cnn_rdkit_feat,
                trans_desc_feat, trans_ecfp_feat, trans_maccs_feat, trans_rdkit_feat,
                gnn_feat
            ], dim=1)
            
            latent_features = model.fusion(combined)
            embeddings.append(latent_features.cpu().numpy())
            
    return np.vstack(embeddings)


def uncertainty_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with highest uncertainty (closest to 0.5 probability)"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # For binary classification, uncertainty is highest when probability is close to 0.5
    uncertainty = np.abs(probs[:, 1] - 0.5)  # Distance from 0.5
    
    # Select samples with the smallest distance to 0.5 (highest uncertainty)
    selected_idx = np.argsort(uncertainty)[-n_samples:]
    return selected_idx


def entropy_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with highest prediction entropy"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    entropy_scores = entropy(probs.T)
    selected_idx = np.argsort(entropy_scores)[-n_samples:]
    return selected_idx


def margin_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with the smallest margin between the top two class probabilities"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # Sort probabilities for each sample and find the margin
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # Select samples with the smallest margin
    selected_idx = np.argsort(margin)[:n_samples]
    return selected_idx


def reconstruction_error_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with the highest reconstruction error."""
    model.eval()
    errors = []
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'], reduction='none')

    with torch.no_grad():
        for batch in pool_loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            smiles_tokens = batch['smiles_tokens'].to(device)
            
            _, _, recon_logits = model(desc, ecfp, maccs, rdkit, graph_data, smiles_tokens)
            
            recon_target = smiles_tokens[:, 1:]
            
            # Calculate loss for each token, then average over the sequence length
            loss = criterion_recon(
                recon_logits.reshape(-1, SMILES_VOCAB_SIZE),
                recon_target.reshape(-1)
            ).reshape(recon_logits.size(0), -1)
            
            # Mask out padding before taking the mean
            mask = (recon_target != char_to_idx['<pad>']).float()
            loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
            errors.append(loss.cpu().numpy())
            
    errors = np.concatenate(errors)
    selected_idx = np.argsort(errors)[-n_samples:]
    return selected_idx


def random_sampling(pool_size: int, n_samples: int) -> np.ndarray:
    """Random sampling"""
    return np.random.choice(pool_size, n_samples, replace=False)


def novelty_sampling(model, pool_dataset, labeled_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples most different from the labeled set in the latent space."""
    pool_embeddings = get_latent_embeddings(model, pool_dataset, device)
    labeled_embeddings = get_latent_embeddings(model, labeled_dataset, device)
    
    # Find the distance of each pool sample to its nearest neighbor in the labeled set
    distances = pairwise_distances(pool_embeddings, labeled_embeddings, metric='euclidean').min(axis=1)
    
    # Select the samples with the largest minimum distances
    selected_idx = np.argsort(distances)[-n_samples:]
    return selected_idx


def diversity_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select diverse samples using k-means++ like approach in the latent space."""
    pool_embeddings = get_latent_embeddings(model, pool_dataset, device)
    
    selected_idx = []
    # Select the first point randomly
    first_idx = np.random.randint(0, len(pool_embeddings))
    selected_idx.append(first_idx)
    
    for _ in range(n_samples - 1):
        selected_features = pool_embeddings[selected_idx]
        # Calculate distance from all points to the already selected points
        distances = pairwise_distances(pool_embeddings, selected_features, metric='euclidean')
        # Find the minimum distance for each point to any of the selected points
        min_distances = distances.min(axis=1)
        
        # Avoid re-selecting already chosen samples
        min_distances[selected_idx] = -1
        # Select the point that is furthest from any already selected point
        next_idx = np.argmax(min_distances)
        selected_idx.append(next_idx)
    
    return np.array(selected_idx)


# === 8. Compute CV AUPRC ===
def compute_cv_auprc(desc, ecfp, maccs, rdkit, smiles, labels, n_splits=5):
    """Compute 5-fold CV AUPRC for Multimodal"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    progress_bar = tqdm(cv.split(np.zeros(len(labels)), labels), total=n_splits, desc="5-Fold CV Progress")
    for train_idx, val_idx in progress_bar:
        # Determine dynamic batch size for this fold
        batch_size = get_dynamic_batch_size(len(train_idx))

        # Create datasets
        train_dataset = MolecularDataset(
            desc.iloc[train_idx], ecfp.iloc[train_idx], maccs.iloc[train_idx], rdkit.iloc[train_idx], smiles[train_idx], labels[train_idx]
        )
        val_dataset = MolecularDataset(
            desc.iloc[val_idx], ecfp.iloc[val_idx], maccs.iloc[val_idx], rdkit.iloc[val_idx], smiles[val_idx], labels[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
        
        # Create and train model
        model = Multimodal(
            desc_dim=desc.shape[1],
            ecfp_dim=ecfp.shape[1],
            maccs_dim=maccs.shape[1],
            rdkit_dim=rdkit.shape[1]
        ).to(device)
        
        criterion_cls = FocalLoss()
        criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Train for a few epochs
        for epoch in range(10):
            train_multimodal(model, train_loader, criterion_cls, criterion_recon, optimizer, device)
        
        # Evaluate
        probs, val_labels = evaluate_multimodal(model, val_loader, device)
        auprc = average_precision_score(val_labels, probs[:, 1])
        cv_scores.append(auprc)
    
    return np.array(cv_scores)


# === 8.6. Generate Reconstructions ===
def generate_reconstructions(model, test_loader, idx_to_char, device, output_dir, n_samples_to_show=20):
    """
    Generates reconstructed SMILES from the test set using the trained autoencoder.
    """
    model.eval()
    original_smiles, reconstructed_smiles, predictions, true_labels = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Generating Reconstructions")):
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            labels = batch['label']
            
            # --- 1. Encoder Pass to get latent features ---
            cnn_desc_feat = model.cnn_desc(desc)
            cnn_ecfp_feat = model.cnn_ecfp(ecfp)
            cnn_maccs_feat = model.cnn_maccs(maccs)
            cnn_rdkit_feat = model.cnn_rdkit(rdkit)
            trans_desc_feat = model.trans_desc(desc)
            trans_ecfp_feat = model.trans_ecfp(ecfp)
            trans_maccs_feat = model.trans_maccs(maccs)
            trans_rdkit_feat = model.trans_rdkit(rdkit)
            gnn_feat = model.gnn(graph_data)
            
            combined = torch.cat([
                cnn_desc_feat, cnn_ecfp_feat, cnn_maccs_feat, cnn_rdkit_feat,
                trans_desc_feat, trans_ecfp_feat, trans_maccs_feat, trans_rdkit_feat,
                gnn_feat
            ], dim=1)
            
            latent_features = model.fusion(combined)
            # --- Get classification prediction ---
            logits = model.classifier(latent_features)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            # --- 2. Autoregressive Decoding ---
            batch_size = latent_features.size(0)
            # Start with the <sos> token for each sequence in the batch
            decoder_input = torch.full((batch_size, 1), char_to_idx['<sos>'], dtype=torch.long, device=device)
            
            # Project the latent features once to create the memory for the decoder
            memory = model.smiles_decoder.latent_to_memory(latent_features)
            
            # Generate sequence step-by-step
            for _ in range(MAX_SMILES_LEN - 1):
                # The memory shape for the decoder should be (seq_len, batch, embed_dim)
                # but since we generate one token at a time, we can adapt.
                # Let's make memory (batch, 1, embed_dim) for simplicity with batch_first=True
                memory_for_step = memory.unsqueeze(1)

                tgt_embed = model.smiles_decoder.embedding(decoder_input) + model.smiles_decoder.pos_encoder[:, :decoder_input.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)

                # The memory needs to be repeated for each token in the target sequence
                memory_proj = memory.unsqueeze(1).repeat(1, decoder_input.size(1), 1)

                output = model.smiles_decoder.transformer_decoder(tgt_embed, memory_proj, tgt_mask=tgt_mask)
                
                # Get the prediction for the very last token
                last_token_logits = model.smiles_decoder.fc_out(output[:, -1, :])
                next_token = torch.argmax(last_token_logits, dim=1).unsqueeze(1)
                
                # Append the predicted token to the input for the next iteration
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # --- 3. Convert tokens to SMILES strings ---
            for j in range(batch_size):
                original_smiles.append(batch['smiles'][j])
                true_labels.append(labels[j].item())
                predictions.append(probs[j])
                
                # Convert sequence of indices to string
                seq = ""
                for token_idx in decoder_input[j, :]:
                    char = idx_to_char.get(token_idx.item())
                    if char == '<eos>': break
                    if char not in ['<sos>', '<pad>']:
                        seq += char
                reconstructed_smiles.append(seq)

    # Create and save DataFrame
    df = pd.DataFrame({
        'Original_SMILES': original_smiles,
        'Reconstructed_SMILES': reconstructed_smiles,
        'True_Label': true_labels,
        'Predicted_Proba': predictions
    })
    
    output_path = os.path.join(output_dir, "baseline_reconstructions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} SMILES reconstructions to {output_path}")
    
    # Print a few examples
    print("\n🔍 Example Reconstructions:")
    print(df.head(n_samples_to_show).to_string())
    
    return df


# === 8.5. Baseline: Train with 100% data ===
def train_baseline_multimodal(
    desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
    desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
    output_dir,
    epochs=20,
    batch_size=32
):
    """Train Multimodal with 100% training data, using a validation split."""
    
    print(f"\n{'='*80}")
    print(f"🎯 Training Baseline Multimodal (100% data with 80/20 validation split)")
    print(f"{'='*80}\n")
    
    # Create datasets
    full_train_dataset = MolecularDataset(desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train)
    test_dataset = MolecularDataset(desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test)
    
    # Use dynamic batch size for baseline training as well, though it's less likely to be small
    batch_size = get_dynamic_batch_size(len(full_train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = Multimodal(
        desc_dim=desc_train.shape[1],
        ecfp_dim=ecfp_train.shape[1],
        maccs_dim=maccs_train.shape[1],
        rdkit_dim=rdkit_train.shape[1]
    ).to(device)
    
    criterion_cls = FocalLoss()
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    print(f"📊 Training on {len(y_train)} samples (split 80/20) for {epochs} epochs")
    print(f"📊 Testing on {len(y_test)} samples\n")
    
    # Train using the validation split to get the best model
    best_model_state, history = train_with_validation(
        model, full_train_dataset, epochs, criterion_cls, criterion_recon, optimizer, scheduler, device
    )

    # Load the best model for final evaluation
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(output_dir, 'best_baseline_model.pth'))
    
    # Generate and save SMILES reconstructions from the test set
    generate_reconstructions(model, test_loader, idx_to_char, device, output_dir)
    
    # Final evaluation on the unseen test set
    test_probs, test_labels = evaluate_multimodal(model, test_loader, device)
    test_preds = np.argmax(test_probs, axis=1)
    
    final_test_auprc = average_precision_score(test_labels, test_probs[:, 1])
    final_test_auc = roc_auc_score(test_labels, test_probs[:, 1])
    final_test_bacc = balanced_accuracy_score(test_labels, test_preds)
    final_test_f1 = f1_score(test_labels, test_preds)

    print(f"\n✅ Best Validation AUPRC during training: {max(history['val_auprc']):.4f}")
    print(f"✅ Final Test AUPRC (from best model): {final_test_auprc:.4f}")
    print(f"✅ Final Test AUC (from best model): {final_test_auc:.4f}")
    print(f"✅ Final Test BACC (from best model): {final_test_bacc:.4f}")
    print(f"✅ Final Test F1 (from best model): {final_test_f1:.4f}")
    
    # Compute 5-Fold CV AUPRC on the full training data
    print(f"\n🔍 Computing 5-Fold CV AUPRC...")
    cv_scores = compute_cv_auprc(desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train, n_splits=5)
    cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
    
    print(f"✅ 5-Fold CV AUPRC: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Save baseline results
    baseline_results = {
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'epochs': epochs,
        'best_val_auprc': max(history['val_auprc']),
        'final_test_auprc': final_test_auprc,
        'final_test_auc': final_test_auc,
        'final_test_bacc': final_test_bacc,
        'final_test_f1': final_test_f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_scores.tolist(),
        'train_losses': history['loss'],
        'train_accs': history['acc'],
        'val_auprcs': history['val_auprc']
    }
    
    # Save to CSV
    baseline_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'train_loss': history['loss'],
        'train_acc': history['acc'],
        'val_auprc': history['val_auprc']
    })
    baseline_csv_path = os.path.join(output_dir, "baseline_training_history.csv")
    baseline_df.to_csv(baseline_csv_path, index=False)
    print(f"\n✅ Saved training history to {baseline_csv_path}")
    
    # Save summary
    summary_df = pd.DataFrame([{
        'model': 'Multimodal_Baseline',
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'best_val_auprc': max(history['val_auprc']),
        'final_test_auprc': final_test_auprc,
        'final_test_auc': final_test_auc,
        'final_test_bacc': final_test_bacc,
        'final_test_f1': final_test_f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }])
    summary_csv_path = os.path.join(output_dir, "baseline_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary to {summary_csv_path}")
    
    # Plot training history
    plot_baseline_training(history['loss'], history['acc'], history['val_auprcs'], output_dir)
    
    return baseline_results


def plot_baseline_training(train_losses, train_accs, val_auprcs, output_dir):
    """Plot baseline training curves"""
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    epochs_range = list(range(1, len(train_losses) + 1))
    
    # Loss curve
    axes[0].plot(epochs_range, train_losses, 'dimgray')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold',style='italic')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold', style='italic')
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold', style='italic')
    axes[0].grid(True, alpha=0.7, linestyle='--')
    
    # Accuracy curve
    axes[1].plot(epochs_range, train_accs, 'goldenrod')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold', style='italic')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', style='italic')
    axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold', style='italic')
    axes[1].grid(True, alpha=0.7, linestyle='--')
    
    # Validation AUPRC curve
    axes[2].plot(epochs_range, val_auprcs, 'royalblue')
    axes[2].axhline(y=max(val_auprcs), color='orange', linestyle='--', label=f'Best: {max(val_auprcs):.4f}')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold', style='italic')
    axes[2].set_ylabel('Validation AUPRC', fontsize=12, fontweight='bold', style='italic')
    axes[2].set_title('Validation AUPRC', fontsize=12, fontweight='bold', style='italic')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[2].grid(True, alpha=0.7, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "baseline_training_curves.svg")
    plt.savefig(output_file, dpi=500, format='svg', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training curves: {output_file}")


# === 8.8. UMAP Visualization Function ===
def plot_umap_sampling(
    all_embeddings_2d, labeled_idx, pool_idx, query_idx,
    strategy_name, round_num, output_dir,
    smiles_train, y_train
):
    """
    Generates and saves a UMAP plot visualizing the sampling process
    and saves the coordinates and status of all points to a CSV.
    """
    
    # --- Plotting ---
    labeled_emb = all_embeddings_2d[labeled_idx]
    pool_emb = all_embeddings_2d[pool_idx]
    query_emb = all_embeddings_2d[query_idx]

    plt.figure(figsize=(3, 3))
    
    # Plot pool data (grey)
    plt.scatter(pool_emb[:, 0], pool_emb[:, 1], c='dimgray', alpha=0.7, label='Pool data')
    
    # Plot labeled data (blue)
    plt.scatter(labeled_emb[:, 0], labeled_emb[:, 1], c='royalblue', alpha=0.7, label='Training data')
    
    # Plot queried data (yellow)
    plt.scatter(query_emb[:, 0], query_emb[:, 1], c='goldenrod', edgecolor='black', linewidth=1, label='Queried data')
    
    plt.title(f'UMAP of Latent Space - Round {round_num}, Strategy: {strategy_name.capitalize()}', fontsize=12, fontweight='bold', style='italic')
    plt.xlabel('UMAP 1', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('UMAP 2', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Create a dedicated folder for UMAP plots
    umap_dir = os.path.join(output_dir, "umap_plots")
    os.makedirs(umap_dir, exist_ok=True)
    
    output_file = os.path.join(umap_dir, f"round_{round_num}_{strategy_name}.svg")
    plt.savefig(output_file, dpi=300, format='svg', bbox_inches='tight')
    plt.close()

    # --- Save Coordinates to CSV ---
    status = np.full(len(all_embeddings_2d), 'pool', dtype=object)
    status[labeled_idx] = 'labeled'
    status[query_idx] = 'queried'

    df = pd.DataFrame({
        'UMAP_1': all_embeddings_2d[:, 0],
        'UMAP_2': all_embeddings_2d[:, 1],
        'SMILES': smiles_train,
        'Label': y_train,
        'Status': status
    })

    # Create a dedicated folder for UMAP data
    umap_data_dir = os.path.join(output_dir, "umap_data")
    os.makedirs(umap_data_dir, exist_ok=True)
    
    csv_output_file = os.path.join(umap_data_dir, f"round_{round_num}_{strategy_name}_coords.csv")
    df.to_csv(csv_output_file, index=False)


# === 9. Active Learning Experiment ===
def active_learning_multimodal(
    desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
    desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
    output_dir,
    n_initial=50,
    n_queries=9,
    n_instances=50,
    epochs_per_round=20
):
    """Run active learning experiment with Multimodal"""
    
    # Initial split
    initial_idx = np.random.choice(len(desc_train), n_initial, replace=False)
    pool_idx = np.setdiff1d(np.arange(len(desc_train)), initial_idx)
    
    # Track performance for test set
    strategies = ['random', 'uncertainty', 'entropy', 'margin', 'novelty', 'diversity']
    metrics = ['auprc', 'auc', 'bacc', 'f1']
    test_performance = {s: {m: [] for m in metrics} for s in strategies}
    sample_sizes = [n_initial]
    
    print(f"\n📊 Initial training set size: {n_initial} samples")
    print(f"📊 Initial pool size: {len(pool_idx)} samples")
    
    # Initial performance
    labeled_dataset = MolecularDataset(
        desc_train.iloc[initial_idx], ecfp_train.iloc[initial_idx],
        maccs_train.iloc[initial_idx], rdkit_train.iloc[initial_idx], smiles_train[initial_idx], y_train[initial_idx]
    )
    test_dataset = MolecularDataset(desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test)
    
    # Train initial model
    model = Multimodal(
        desc_dim=desc_train.shape[1],
        ecfp_dim=ecfp_train.shape[1],
        maccs_dim=maccs_train.shape[1],
        rdkit_dim=rdkit_train.shape[1]
    ).to(device)
    
    initial_batch_size = get_dynamic_batch_size(len(labeled_dataset))
    train_loader = DataLoader(labeled_dataset, batch_size=initial_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    criterion_cls = FocalLoss()
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    print(f"\n🎯 Training initial model...")
    best_initial_state, _ = train_with_validation(
        model, labeled_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
    )
    model.load_state_dict(best_initial_state)
    
    # Initial test metrics
    probs, labels = evaluate_multimodal(model, test_loader, device)
    preds = np.argmax(probs, axis=1)
    initial_metrics = {
        'auprc': average_precision_score(labels, probs[:, 1]),
        'auc': roc_auc_score(labels, probs[:, 1]),
        'bacc': balanced_accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }
    
    print(f"\n🎯 Query round 0/{n_queries}")
    print(f"Initial performance:")
    for strategy in strategies:
        for metric, value in initial_metrics.items():
            test_performance[strategy][metric].append(value)
        print(f"  {strategy.capitalize()}: Test AUPRC={initial_metrics['auprc']:.4f}, Test AUC={initial_metrics['auc']:.4f}, Test BACC={initial_metrics['bacc']:.4f}, Test F1={initial_metrics['f1']:.4f}")
    
    # Active learning loop
    current_labeled_idx = initial_idx.copy()
    current_pool_idx = pool_idx.copy();
    
    for i in range(n_queries):
        print(f"\n🎯 Query round {i+1}/{n_queries}")
        
        if len(current_pool_idx) == 0:
            print("⚠️  No more samples in pool. Stopping early.")
            break
        
        n_instances_round = min(n_instances, len(current_pool_idx))
        print(f"📊 Current training set size: {len(current_labeled_idx)} samples")
        print(f"📊 Remaining pool size: {len(current_pool_idx)} samples")
        print(f"📊 Selecting {n_instances_round} samples this round")
        
        selected_indices = {}
        
        # --- UMAP Setup for this round ---
        # To ensure consistent projections, we fit UMAP on the embeddings of the *entire* dataset
        # using a model trained on the current labeled set.
        print("\n  ⚙️  Fitting UMAP for visualization...")
        temp_model = Multimodal(
            desc_dim=desc_train.shape[1], ecfp_dim=ecfp_train.shape[1],
            maccs_dim=maccs_train.shape[1], rdkit_dim=rdkit_train.shape[1]
        ).to(device)
        
        temp_batch_size = get_dynamic_batch_size(len(current_labeled_idx))
        temp_train_loader = DataLoader(
            MolecularDataset(
                desc_train.iloc[current_labeled_idx], ecfp_train.iloc[current_labeled_idx],
                maccs_train.iloc[current_labeled_idx], rdkit_train.iloc[current_labeled_idx],
                smiles_train[current_labeled_idx], y_train[current_labeled_idx]
            ),
            batch_size=temp_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
        )
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
        temp_criterion_cls = FocalLoss()
        temp_criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
        
        # Train a temporary model just for this round's embeddings
        best_temp_state, _ = train_with_validation(
            temp_model, temp_train_loader.dataset, epochs_per_round, temp_criterion_cls, temp_criterion_recon, temp_optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(temp_optimizer), device
        )
        temp_model.load_state_dict(best_temp_state)
        
        full_dataset_for_umap = MolecularDataset(desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train)
        all_embeddings = get_latent_embeddings(temp_model, full_dataset_for_umap, device)
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(all_embeddings)
        all_embeddings_2d = umap_reducer.transform(all_embeddings)
        # --- End UMAP Setup ---

        for strategy in strategies:
            print(f"\n  🔍 Strategy: {strategy.capitalize()}")
            
            # Create current datasets
            current_labeled_dataset = MolecularDataset(
                desc_train.iloc[current_labeled_idx], ecfp_train.iloc[current_labeled_idx],
                maccs_train.iloc[current_labeled_idx], rdkit_train.iloc[current_labeled_idx],
                smiles_train[current_labeled_idx], y_train[current_labeled_idx]
            )
            current_pool_dataset = MolecularDataset(
                desc_train.iloc[current_pool_idx], ecfp_train.iloc[current_pool_idx],
                maccs_train.iloc[current_pool_idx], rdkit_train.iloc[current_pool_idx],
                smiles_train[current_pool_idx], y_train[current_pool_idx] # Pass labels for reconstruction error calculation
            )
            
            # Train model on current labeled set
            model = Multimodal(
                desc_dim=desc_train.shape[1], ecfp_dim=ecfp_train.shape[1],
                maccs_dim=maccs_train.shape[1], rdkit_dim=rdkit_train.shape[1]
            ).to(device)
            
            strategy_batch_size = get_dynamic_batch_size(len(current_labeled_dataset))
            train_loader = DataLoader(current_labeled_dataset, batch_size=strategy_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
            criterion_cls = FocalLoss()
            criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            
            # Train model on current labeled set with validation split
            best_state, _ = train_with_validation(
                model, current_labeled_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
            )
            model.load_state_dict(best_state)
            
            # Select samples
            if strategy == 'random':
                selected_idx = random_sampling(len(current_pool_idx), n_instances_round)
            elif strategy == 'uncertainty':
                selected_idx = uncertainty_sampling(model, current_pool_dataset, n_instances_round, device)
            elif strategy == 'entropy':
                selected_idx = entropy_sampling(model, current_pool_dataset, n_instances_round, device)
            elif strategy == 'margin':
                selected_idx = margin_sampling(model, current_pool_dataset, n_instances_round, device)
            elif strategy == 'novelty':
                selected_idx = novelty_sampling(model, current_pool_dataset, current_labeled_dataset, n_instances_round, device)
            elif strategy == 'diversity':
                selected_idx = diversity_sampling(model, current_pool_dataset, n_instances_round, device)
            
            # --- Save queried SMILES to CSV ---
            queried_smiles_indices = current_pool_idx[selected_idx]
            queried_smiles = smiles_train[queried_smiles_indices]
            smiles_df = pd.DataFrame({'SMILES': queried_smiles})
            
            smiles_dir = os.path.join(output_dir, "queried_smiles")
            os.makedirs(smiles_dir, exist_ok=True)
            
            smiles_filename = os.path.join(smiles_dir, f"round_{i+1}_{strategy}_smiles.csv")
            smiles_df.to_csv(smiles_filename, index=False)
            print(f"    💾 Saved {len(queried_smiles)} queried SMILES to {smiles_filename}")
            # --- End SMILES saving ---

            selected_indices[strategy] = selected_idx
            
            # --- Generate and save UMAP plot for this strategy ---
            plot_umap_sampling(
                all_embeddings_2d=all_embeddings_2d,
                labeled_idx=current_labeled_idx,
                pool_idx=current_pool_idx,
                query_idx=current_pool_idx[selected_idx],
                strategy_name=strategy,
                round_num=i + 1,
                output_dir=output_dir,
                smiles_train=smiles_train,
                y_train=y_train
            )
            print(f"    📊 UMAP plot and coordinates saved for {strategy} strategy.")
            
            # Create strategy-specific dataset
            strategy_labeled_idx = np.concatenate([current_labeled_idx, current_pool_idx[selected_idx]])
            strategy_dataset = MolecularDataset(
                desc_train.iloc[strategy_labeled_idx], ecfp_train.iloc[strategy_labeled_idx],
                maccs_train.iloc[strategy_labeled_idx], rdkit_train.iloc[strategy_labeled_idx],
                smiles_train[strategy_labeled_idx], y_train[strategy_labeled_idx])
            # Train on updated set
            model_final = Multimodal(
                desc_dim=desc_train.shape[1], ecfp_dim=ecfp_train.shape[1],
                maccs_dim=maccs_train.shape[1], rdkit_dim=rdkit_train.shape[1]
            ).to(device)
            optimizer_final = torch.optim.Adam(model_final.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler_final = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='max', factor=0.5, patience=3)

            best_final_state, _ = train_with_validation(
                model_final, strategy_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer_final, scheduler_final, device
            )
            model_final.load_state_dict(best_final_state)

            # Save the final model on the last query round
            if i == n_queries - 1:
                model_dir = os.path.join(output_dir, "final_al_models")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"final_model_{strategy}.pth")
                torch.save(best_final_state, model_path)
                print(f"    💾 Saved final model for {strategy} to {model_path}")

            # Calculate all test metrics
            probs, labels = evaluate_multimodal(model_final, test_loader, device)
            preds = np.argmax(probs, axis=1)
            
            test_metrics = {
                'auprc': average_precision_score(labels, probs[:, 1]),
                'auc': roc_auc_score(labels, probs[:, 1]),
                'bacc': balanced_accuracy_score(labels, preds),
                'f1': f1_score(labels, preds)
            }
            
            for metric, value in test_metrics.items():
                test_performance[strategy][metric].append(value)
            
            print(f"    Test AUPRC={test_metrics['auprc']:.4f}, AUC={test_metrics['auc']:.4f}, BACC={test_metrics['bacc']:.4f}, F1={test_metrics['f1']:.4f}")
        
        # Update for next round (using random selection)
        chosen = selected_indices['random']
        current_labeled_idx = np.concatenate([current_labeled_idx, current_pool_idx[chosen]])
        current_pool_idx = np.delete(current_pool_idx, chosen)
        sample_sizes.append(len(current_labeled_idx))
    
    return test_performance, sample_sizes


# === 10. Plot Learning Curves ===
def plot_learning_curves_separate(
    test_performance,
    sample_sizes,
    total_train_size,
    output_dir,
    model_name="Multimodal"
):
    """Plot separate figures for test metrics with x-axis = % of FULL training set."""
    
    colors = {'random': 'royalblue', 'uncertainty': 'red', 'entropy': 'green',
              'margin': 'purple', 'novelty': 'orange', 'diversity': 'brown'}
    markers = {'random': 'o', 'uncertainty': '^', 'entropy': 'p',
               'margin': 's', 'novelty': 'h', 'diversity': '*'}
    
    # Percent relative to FULL training set (e.g. 250 / 533 ≈ 46.9%)
    sample_percent = [100.0 * s / total_train_size for s in sample_sizes]
    
    # --- Test Metrics Plots ---
    for metric in ['auprc', 'auc', 'bacc', 'f1']:
        metric_upper = metric.upper()
        fig, ax = plt.subplots(figsize=(6,3))
        for strategy, scores_dict in test_performance.items():
            ax.plot(sample_percent, scores_dict[metric], marker=markers[strategy], color=colors[strategy],
                     label=strategy.capitalize(), linewidth=1)
        
        ax.set_xlabel('Training set (% of full train data)', fontsize=12, fontweight='bold', style='italic')
        ax.set_ylabel(f'Test {metric_upper}', fontsize=12, fontweight='bold', style='italic')
        ax.set_title(f'Test Set {metric_upper} - {model_name}', fontsize=12, fontweight='bold', style='italic')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.7, linestyle='--')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f"learning_curves_test_{metric}.svg")
        plt.savefig(output_file, dpi=500, format='svg', bbox_inches='tight')
        plt.close()
        print(f"✅ Saved Test {metric_upper} plot: {output_file}")


# === 11. Main Function ===
def main(endpoint= "hepatotoxicity"):
    print(f"\n{'='*80}")
    print(f"🚀 Starting Active Learning with Multimodal")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading data...")
    
    # Load labels from the data/AL directory
    df = pd.read_excel("Supporting_Information_2.xlsx", sheet_name=f"{endpoint}", index_col=0)
    output_dir = f"active_learning_multimodal_{endpoint}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Folder '{output_dir}' ถูกสร้างแล้ว (หรือมีอยู่แล้ว)")
    train_df = df[df['validation'] == 'training']
    test_df  = df[df['validation'] == 'test']
    y_train = train_df['Label'].values
    y_test  =  test_df['Label'].values
    smiles_train = train_df['canonical_smiles'].values
    smiles_test  = test_df['canonical_smiles'].values
    print(f"✅ Loaded labels and SMILES: {len(y_train)} train, {len(y_test)} test")

    # Calculate features
    train_ecfp = calculate_ecfp(train_df, smiles_col="canonical_smiles", radius=10, nBits=4096)
    test_ecfp  = calculate_ecfp(test_df, smiles_col="canonical_smiles", radius=10, nBits=4096)
    train_maccs = calculate_maccs(train_df, smiles_col="canonical_smiles")
    test_maccs  = calculate_maccs(test_df, smiles_col="canonical_smiles")
    train_rdkit = calculate_rdkit(train_df, smiles_col="canonical_smiles")
    test_rdkit  = calculate_rdkit(test_df, smiles_col="canonical_smiles")
    train_desc  = calculate_descriptors(train_df, smiles_col="canonical_smiles")
    test_desc   = calculate_descriptors(test_df, smiles_col="canonical_smiles")
    print(f"✅ Computed ECFP, MACCS, and RDKit features.")
    
    X_train = {
        'DESC': train_desc,
        'ECFP': train_ecfp,
        'MACCS': train_maccs,
        'RDKIT': train_rdkit
    }
    X_test = {
        'DESC': test_desc,
        'ECFP': test_ecfp,
        'MACCS': test_maccs,
        'RDKIT': test_rdkit
    }
    
    desc_train, ecfp_train, maccs_train, rdkit_train = X_train['DESC'], X_train['ECFP'], X_train['MACCS'], X_train['RDKIT']
    desc_test, ecfp_test, maccs_test, rdkit_test = X_test['DESC'], X_test['ECFP'], X_test['MACCS'], X_test['RDKIT']
    
    print(f"\n📊 Training samples: {len(y_train)}")
    print(f"📊 Test samples: {len(y_test)}")
    
    # === BASELINE: Train with 100% data ===
    baseline_results = train_baseline_multimodal(
        desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
        desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
        output_dir,
        epochs=20,
        batch_size=32
    )
    
    # === ACTIVE LEARNING ===
    print(f"\n{'='*80}")
    print(f"🎯 Starting Active Learning Experiments")
    print(f"{'='*80}")
    
    # Calculate initial and instance sizes based on percentages
    total_train_size = len(y_train)
    n_initial = int(total_train_size * 0.05)
    n_instances = max(1, int(total_train_size * 0.005)) # Ensure at least 1 sample is added
    n_queries = 20 # Number of rounds to run

    print(f"📊 Active Learning Setup:")
    print(f"  - Initial training size: {n_initial} samples (5%)")
    print(f"  - Samples per query: {n_instances} samples (0.5%)")
    print(f"  - Number of queries: {n_queries}")

    # Run active learning
    test_perf, sample_sizes = active_learning_multimodal(
        desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
        desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
        output_dir,
        n_initial=n_initial,
        n_queries=n_queries,
        n_instances=n_instances,
        epochs_per_round=20
    )
    
    # Save active learning results
    df = pd.DataFrame({"n_samples": sample_sizes})
    total_train_size = len(y_train)
    df["percent_of_full_train"] = [100.0 * s / total_train_size for s in sample_sizes]
    
    # Add columns for each metric and strategy
    for strategy, metric_dict in test_perf.items():
        for metric, scores in metric_dict.items():
            df[f"{strategy}_test_{metric}"] = scores
            
    csv_path = os.path.join(output_dir, "performance_active_learning.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved active learning performance to {csv_path}")
    
    # Plot learning curves with full train size reference
    plot_learning_curves_separate(test_perf, sample_sizes, total_train_size, output_dir)

    # === COMPARISON SUMMARY ===
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON: Baseline vs Active Learning")
    print(f"{'='*80}\n")
    
    print(f"BASELINE (100% data):")
    print(f"  Test AUPRC: {baseline_results['final_test_auprc']:.4f}")
    print(f"  Test AUC:   {baseline_results['final_test_auc']:.4f}")
    print(f"  Test BACC:  {baseline_results['final_test_bacc']:.4f}")
    print(f"  Test F1:    {baseline_results['final_test_f1']:.4f}")
    print(f"  5-CV AUPRC: {baseline_results['cv_mean']:.4f} ± {baseline_results['cv_std']:.4f}")
    
    print(f"\nACTIVE LEARNING (Final results):")
    for strategy in test_perf:
        final_metrics = {m: v[-1] for m, v in test_perf[strategy].items()}
        samples_used = sample_sizes[-1]
        percent_used = 100.0 * samples_used / len(y_train)
        print(f"  {strategy.capitalize():12s} ({percent_used:.1f}% data): Test AUPRC={final_metrics['auprc']:.4f}, AUC={final_metrics['auc']:.4f}, BACC={final_metrics['bacc']:.4f}, F1={final_metrics['f1']:.4f}")
    
    # Create comparison table
    comparison_data = [{
        'Method': 'Baseline (100%)',
        'Samples': len(y_train),
        'Percent': 100.0,
        'Test_AUPRC': baseline_results['final_test_auprc'],
        'Test_AUC': baseline_results['final_test_auc'],
        'Test_BACC': baseline_results['final_test_bacc'],
        'Test_F1': baseline_results['final_test_f1'],
        'CV_AUPRC_Mean': baseline_results['cv_mean'],
        'CV_AUPRC_Std': baseline_results['cv_std']
    }]
    
    for strategy in test_perf:
        final_metrics = {m: v[-1] for m, v in test_perf[strategy].items()}
        comparison_data.append({
            'Method': f'AL_{strategy.capitalize()}',
            'Samples': sample_sizes[-1],
            'Percent': 100.0 * sample_sizes[-1] / len(y_train),
            'Test_AUPRC': final_metrics['auprc'],
            'Test_AUC': final_metrics['auc'],
            'Test_BACC': final_metrics['bacc'],
            'Test_F1': final_metrics['f1'],
            'CV_AUPRC_Mean': None, # No CV for AL iterations
            'CV_AUPRC_Std': None
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(output_dir, "comparison_baseline_vs_al.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n✅ Saved comparison to {comparison_csv_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ Multimodal {endpoint} Completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    for endpoint in ["neurotoxicity", "nephrotoxicity", "PBMC_toxicity", "SCARs", "Skinsensitzation"]:
        main(endpoint=endpoint) #"hepatotoxicity", "respirotoxicity", "cardiotoxicity",