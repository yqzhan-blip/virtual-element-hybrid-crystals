"""
Comprehensive organic cation library for hybrid perovskite virtual element mapping.

Sources:
  [1] Kim et al., Scientific Data (2017) - 16 HOIP cations
  [2] Marchenko et al., Chem. Mater. (2020) - 2D hybrid perovskite database
  [3] Various literature on 2D/3D perovskite solar cells

Total cations: 154
  - A-site cations (small, for 3D perovskites): 44
  - Spacer cations (large, for 2D/Ruddlesden-Popper): 110
"""

# ===========================================================================
# A-SITE CATIONS (3D perovskite, radius ~2.0-3.5 Å)
# ===========================================================================

A_SITE_CATIONS = {
    # === Simple ammonium ===
    "AM":  {"smiles": "[NH4+]",         "name": "Ammonium"},
    "MA":  {"smiles": "C[NH3+]",        "name": "Methylammonium"},
    "DMA": {"smiles": "C[NH2+]C",       "name": "Dimethylammonium"},
    "TMA": {"smiles": "C[NH+](C)C",     "name": "Trimethylammonium"},
    "TetraMA": {"smiles": "C[N+](C)(C)C", "name": "Tetramethylammonium"},
    "EA":  {"smiles": "CC[NH3+]",       "name": "Ethylammonium"},
    "PA":  {"smiles": "CCC[NH3+]",      "name": "Propylammonium"},
    "IPA": {"smiles": "CC([NH3+])C",    "name": "Isopropylammonium"},
    "IBA": {"smiles": "CC(C)C[NH3+]",   "name": "Isobutylammonium"},
    "TBA": {"smiles": "CC(C)(C)[NH3+]", "name": "tert-Butylammonium"},

    # === Hydroxyl / ether ammoniums ===
    "HA":  {"smiles": "O[NH3+]",        "name": "Hydroxylammonium"},
    "HYA": {"smiles": "O[NH3+]",        "name": "Hydroxylammonium"},
    "HOEA": {"smiles": "OCC[NH3+]",      "name": "2-Hydroxyethylammonium"},
    "HOPA": {"smiles": "OCCC[NH3+]",     "name": "3-Hydroxypropylammonium"},
    "MOPA": {"smiles": "COCC[NH3+]",    "name": "2-Methoxyethylammonium"},

    # === Amidinium / guanidinium family ===
    "FA":  {"smiles": "C(=[NH2+])N",    "name": "Formamidinium"},
    "AA":  {"smiles": "CC(=[NH2+])N",   "name": "Acetamidinium"},
    "GA":  {"smiles": "NC(=[NH2+])N",   "name": "Guanidinium"},
    "AGA": {"smiles": "NC(=[NH2+])NC",  "name": "N-Methylguanidinium"},
    "DGA": {"smiles": "CN=C(N)N",       "name": "N,N'-Dimethylguanidinium"},

    # === Cyclic ammoniums (small rings) ===
    "AZ":  {"smiles": "C1C[NH2+]C1",    "name": "Azetidinium"},
    "PYR": {"smiles": "C1CC[NH2+]C1",   "name": "Pyrrolidinium"},
    "PIP": {"smiles": "C1CC[NH2+]CC1",  "name": "Piperidinium"},
    "IM":  {"smiles": "[NH2+]1C=CN=C1",   "name": "Imidazolium"},
    "PYZ": {"smiles": "C1=C[NH2+]C=C1", "name": "Pyridinium"},
    "MIM": {"smiles": "Cn1c[nH+]cc1",   "name": "N-Methylimidazolium"},

    # === Hydrazinium ===
    "HZ":  {"smiles": "N[NH3+]",        "name": "Hydrazinium"},
    "MHZ": {"smiles": "CN[NH3+]",       "name": "Methylhydrazinium"},

    # === Halogenated ethylammoniums ===
    "FEA": {"smiles": "FCC[NH3+]",      "name": "2-Fluoroethylammonium"},
    "ClEA":{"smiles": "ClCC[NH3+]",     "name": "2-Chloroethylammonium"},
    "BrEA":{"smiles": "BrCC[NH3+]",     "name": "2-Bromoethylammonium"},

    # === Small diamines (A-site capable) ===
    "EDA": {"smiles": "C(C[NH3+])[NH3+]",  "name": "Ethylenediammonium"},
    "PDA": {"smiles": "C(CC[NH3+])[NH3+]", "name": "1,3-Propanediammonium"},
    "BDA": {"smiles": "C(CCC[NH3+])[NH3+]","name": "1,4-Butanediammonium"},

    # === Unsaturated ===
    "AL":  {"smiles": "C=C[NH3+]",      "name": "Allylammonium"},
    "PRG": {"smiles": "C#CC[NH3+]",     "name": "Propargylammonium"},

    # === Cyano / nitro ===
    "CYA": {"smiles": "N#CC[NH3+]",     "name": "Cyanomethylammonium"},
    "NTA": {"smiles": "O=N(=O)C[NH3+]", "name": "Nitromethylammonium"},
    "APN": {"smiles": "N#CCC[NH3+]",    "name": "3-Aminopropionitrile"},
}

# ===========================================================================
# SPACER CATIONS (2D perovskite, radius ~3.5-10+ Å)
# ===========================================================================

SPACER_CATIONS = {
    # === Linear alkylammoniums (CnH2n+1NH3+, n=4-18) ===
    "BA":  {"smiles": "CCCC[NH3+]",        "name": "n-Butylammonium"},
    "PA5": {"smiles": "CCCCC[NH3+]",       "name": "n-Pentylammonium"},
    "HA":  {"smiles": "CCCCCC[NH3+]",      "name": "n-Hexylammonium"},
    "HA7": {"smiles": "CCCCCCC[NH3+]",     "name": "n-Heptylammonium"},
    "OA":  {"smiles": "CCCCCCCC[NH3+]",    "name": "n-Octylammonium"},
    "NA":  {"smiles": "CCCCCCCCC[NH3+]",   "name": "n-Nonylammonium"},
    "DA":  {"smiles": "CCCCCCCCCC[NH3+]",  "name": "n-Decylammonium"},
    "UDA": {"smiles": "CCCCCCCCCCC[NH3+]", "name": "n-Undecylammonium"},
    "DDA": {"smiles": "CCCCCCCCCCCC[NH3+]","name": "n-Dodecylammonium"},
    "TDA": {"smiles": "CCCCCCCCCCCCCC[NH3+]","name": "n-Tetradecylammonium"},
    "HDA": {"smiles": "CCCCCCCCCCCCCCCC[NH3+]","name": "n-Hexadecylammonium"},
    "ODA": {"smiles": "CCCCCCCCCCCCCCCCCC[NH3+]","name": "n-Octadecylammonium"},

    # === Branched alkylammoniums ===
    "IBA2": {"smiles": "CCCC(C)[NH3+]",       "name": "2-Methylbutylammonium"},
    "EHA":  {"smiles": "CCCCC(CC)C[NH3+]",    "name": "2-Ethylhexylammonium"},
    "DMA2": {"smiles": "CCCCCCCC(C)[NH3+]",   "name": "2-Methyloctylammonium"},

    # === Phenethylammonium series (PEA derivatives) ===
    "PEA": {"smiles": "NCCC1=CC=CC=C1",        "name": "Phenethylammonium"},
    "4F-PEA": {"smiles": "NCCC1=CC=C(F)C=C1",  "name": "4-Fluorophenethylammonium"},
    "3F-PEA": {"smiles": "NCCC1=CC(F)=CC=C1",  "name": "3-Fluorophenethylammonium"},
    "2F-PEA": {"smiles": "NCCC1=C(F)C=CC=C1",  "name": "2-Fluorophenethylammonium"},
    "4Cl-PEA": {"smiles": "NCCC1=CC=C(Cl)C=C1","name": "4-Chlorophenethylammonium"},
    "4Br-PEA": {"smiles": "NCCC1=CC=C(Br)C=C1","name": "4-Bromophenethylammonium"},
    "4I-PEA":  {"smiles": "NCCC1=CC=C(I)C=C1", "name": "4-Iodophenethylammonium"},
    "4Me-PEA": {"smiles": "NCCC1=CC=C(C)C=C1", "name": "4-Methylphenethylammonium"},
    "4OMe-PEA": {"smiles": "NCCC1=CC=C(OC)C=C1","name": "4-Methoxyphenethylammonium"},
    "4OH-PEA": {"smiles": "NCCC1=CC=C(O)C=C1", "name": "4-Hydroxyphenethylammonium"},
    "4CN-PEA": {"smiles": "NCCC1=CC=C(C#N)C=C1","name": "4-Cyanophenethylammonium"},
    "4NO2-PEA": {"smiles": "NCCC1=CC=C([N+](=O)[O-])C=C1","name": "4-Nitrophenethylammonium"},
    "4NH2-PEA": {"smiles": "NCCC1=CC=C(N)C=C1","name": "4-Aminophenethylammonium"},
    "3Me-4F-PEA": {"smiles": "NCCC1=CC(C)=C(F)C=C1","name": "3-Methyl-4-fluorophenethylammonium"},
    "34F2-PEA": {"smiles": "NCCC1=CC(F)=C(F)C=C1","name": "3,4-Difluorophenethylammonium"},
    "345F3-PEA": {"smiles": "NCCC1=CC(F)=C(F)C(F)=C1","name": "3,4,5-Trifluorophenethylammonium"},
    "35F2-PEA": {"smiles": "NCCC1=CC(F)=CC(F)=C1","name": "3,5-Difluorophenethylammonium"},
    "25F2-PEA": {"smiles": "NCCC1=C(F)C=CC(F)=C1","name": "2,5-Difluorophenethylammonium"},
    "26F2-PEA": {"smiles": "NCCC1=C(F)C=CC=C1F","name": "2,6-Difluorophenethylammonium"},
    "24F2-PEA": {"smiles": "NCCC1=C(F)C=C(F)C=C1","name": "2,4-Difluorophenethylammonium"},
    "PentaF-PEA": {"smiles": "NCCC1=C(F)C(F)=C(F)C(F)=C1F","name": "Pentafluorophenethylammonium"},

    # === Benzylammonium series ===
    "BZA": {"smiles": "NCC1=CC=CC=C1",         "name": "Benzylammonium"},
    "4F-BZA": {"smiles": "NCC1=CC=C(F)C=C1",   "name": "4-Fluorobenzylammonium"},
    "3F-BZA": {"smiles": "NCC1=CC(F)=CC=C1",   "name": "3-Fluorobenzylammonium"},
    "4Cl-BZA": {"smiles": "NCC1=CC=C(Cl)C=C1", "name": "4-Chlorobenzylammonium"},
    "4Br-BZA": {"smiles": "NCC1=CC=C(Br)C=C1", "name": "4-Bromobenzylammonium"},
    "4Me-BZA": {"smiles": "NCC1=CC=C(C)C=C1",  "name": "4-Methylbenzylammonium"},
    "4OMe-BZA": {"smiles": "NCC1=CC=C(OC)C=C1","name": "4-Methoxybenzylammonium"},
    "34F2-BZA": {"smiles": "NCC1=CC(F)=C(F)C=C1","name": "3,4-Difluorobenzylammonium"},

    # === Phenylpropyl / phenylbutyl ammoniums ===
    "PPA": {"smiles": "NCCCC1=CC=CC=C1",         "name": "3-Phenylpropylammonium"},
    "PBA": {"smiles": "NCCCCC1=CC=CC=C1",         "name": "4-Phenylbutylammonium"},
    "4F-PPA": {"smiles": "NCCCC1=CC=C(F)C=C1",   "name": "3-(4-Fluorophenyl)propylammonium"},
    "4F-PBA": {"smiles": "NCCCCC1=CC=C(F)C=C1",  "name": "4-(4-Fluorophenyl)butylammonium"},

    # === Naphthalene-based ===
    "NEA":  {"smiles": "NCCC1=C2C=CC=CC2=CC=C1",   "name": "1-Naphthylethylammonium"},
    "NMA":  {"smiles": "NCC1=C2C=CC=CC2=CC=C1",     "name": "1-Naphthylmethylammonium"},
    "2-NEA": {"smiles": "NCCC1=CC=C2C=CC=CC2=C1",   "name": "2-Naphthylethylammonium"},
    "2-NMA": {"smiles": "NCC1=CC=C2C=CC=CC2=C1",    "name": "2-Naphthylmethylammonium"},
    "6F-NEA": {"smiles": "NCCC1=C2C=CC(F)=CC2=CC=C1","name": "2-(6-Fluoro-1-naphthyl)ethylammonium"},

    # === Biphenyl / terphenyl ===
    "BPEA": {"smiles": "NCCC1=CC=C(C2=CC=CC=C2)C=C1","name": "4-Biphenylethylammonium"},
    "BPMA": {"smiles": "NCC1=CC=C(C2=CC=CC=C2)C=C1", "name": "4-Biphenylmethylammonium"},

    # === Anthracene / pyrene ===
    "ANEA": {"smiles": "NCCC1=C2C=C3C=CC=CC3=CC2=CC=C1","name": "1-Anthrylethylammonium"},
    "PYEA": {"smiles": "NCCC1=C2C=CC3=CC=CC4=CC=C(C1)C2=C34","name": "1-Pyrenylethylammonium"},

    # === Thiophene-based ===
    "TEA":  {"smiles": "NCCC1=CC=CS1",        "name": "2-Thienylethylammonium"},
    "TMA2": {"smiles": "NCC1=CC=CS1",          "name": "2-Thienylmethylammonium"},
    "BTEA": {"smiles": "NCCC1=CC2=C(C=CS2)S1", "name": "2-(Benzothienyl)ethylammonium"},
    "TTPA": {"smiles": "NCCCC1=CC=C(C2=CC=CS2)S1","name": "2-(5'-(Thien-2-yl)-2,2'-bithien-5-yl)ethylammonium"},

    # === Furan / pyridine ===
    "FEA":  {"smiles": "NCCC1=CC=CO1",         "name": "2-Furylethylammonium"},
    "PyEA": {"smiles": "NCCC1=CC=NC=C1",       "name": "2-(4-Pyridyl)ethylammonium"},
    "PyMA": {"smiles": "NCC1=CC=NC=C1",        "name": "4-Pyridylmethylammonium"},
    "2PyEA":{"smiles": "NCCC1=NC=CC=C1",       "name": "2-(2-Pyridyl)ethylammonium"},
    "3PyEA":{"smiles": "NCCC1=CN=CC=C1",       "name": "2-(3-Pyridyl)ethylammonium"},

    # === Cyclohexyl / cyclopentyl ===
    "CHA":  {"smiles": "NC1CCCCC1",             "name": "Cyclohexylammonium"},
    "CPA":  {"smiles": "NC1CCCC1",              "name": "Cyclopentylammonium"},
    "CHMA": {"smiles": "NCC1CCCCC1",            "name": "Cyclohexylmethylammonium"},
    "CHEA": {"smiles": "NCCC1CCCCC1",           "name": "2-Cyclohexylethylammonium"},

    # === Adamantyl ===
    "ADA": {"smiles": "NC12CC3CC(CC(C3)C1)C2",  "name": "1-Adamantylammonium"},
    "ADMA":{"smiles": "NCC12CC3CC(CC(C3)C1)C2", "name": "1-Adamantylmethylammonium"},
    "ADEA":{"smiles": "NCCC12CC3CC(CC(C3)C1)C2","name": "2-(1-Adamantyl)ethylammonium"},

    # === Functionalized spacer cations (diammonium) ===
    "BDA2": {"smiles": "C(CCC[NH3+])[NH3+]",   "name": "1,4-Butanediammonium (spacer)"},
    "PDA2": {"smiles": "C(CCCCC[NH3+])[NH3+]",  "name": "1,5-Pentanediammonium"},
    "HDA2": {"smiles": "C(CCCCCC[NH3+])[NH3+]", "name": "1,6-Hexanediammonium"},
    "ODA2": {"smiles": "C(CCCCCCCC[NH3+])[NH3+]","name": "1,8-Octanediammonium"},
    "DDA2": {"smiles": "C(CCCCCCCCCC[NH3+])[NH3+]","name": "1,10-Decanediammonium"},
    "DDDA2":{"smiles": "C(CCCCCCCCCCCC[NH3+])[NH3+]","name": "1,12-Dodecanediammonium"},

    # === Xylylenediammonium (芳香二铵) ===
    "pXDA": {"smiles": "C1=CC(=CC=C1C[NH3+])C[NH3+]","name": "p-Xylylenediammonium"},
    "mXDA": {"smiles": "C1=CC(=CC(=C1)C[NH3+])C[NH3+]","name": "m-Xylylenediammonium"},
    "oXDA": {"smiles": "C1=CC(=C(C=C1)C[NH3+])C[NH3+]","name": "o-Xylylenediammonium"},

    # === Fluoroalkyl ammoniums ===
    "FBA":  {"smiles": "FCCCC[NH3+]",          "name": "4-Fluorobutylammonium"},
    "FHA":  {"smiles": "FCCCCCC[NH3+]",        "name": "6-Fluorohexylammonium"},
    "FOA":  {"smiles": "FCCCCCCCC[NH3+]",      "name": "8-Fluorooctylammonium"},
    "DFBA": {"smiles": "FCCCC(F)[NH3+]",       "name": "4,4-Difluorobutylammonium"},

    # === Oligo(ethylene glycol) ammoniums ===
    "OEG1": {"smiles": "COCCOCC[NH3+]",        "name": "2-(2-Methoxyethoxy)ethylammonium"},
    "OEG2": {"smiles": "COCCOCCOCC[NH3+]",     "name": "2-[2-(2-Methoxyethoxy)ethoxy]ethylammonium"},
    "OEG3": {"smiles": "COCCOCCOCCOCC[NH3+]",  "name": "2-[2-[2-(2-Methoxyethoxy)ethoxy]ethoxy]ethylammonium"},

    # === Ammonium-terminated siloxanes ===
    "AMPS": {"smiles": "C[Si](C)(C)CC[NH3+]",   "name": "3-(Trimethylsilyl)propylammonium"},

    # === Cyano-functionalized alkyl ===
    "CNBA": {"smiles": "N#CCCCCC[NH3+]",       "name": "6-Cyanohexylammonium"},
    "CNOA": {"smiles": "N#CCCCCCCC[NH3+]",     "name": "8-Cyanooctylammonium"},

    # === Ester-functionalized ===
    "AEBA": {"smiles": "CCOC(=O)CC[NH3+]",     "name": "Ethyl 3-ammoniopropionate"},
    "MEHA": {"smiles": "COC(=O)CCCCC[NH3+]",   "name": "Methyl 6-ammoniohexanoate"},

    # === Unsaturated spacer ===
    "OLE": {"smiles": "CCCCCCCC/C=C\\CCCCCCCC[NH3+]","name": "Oleylammonium"},
    "CINA": {"smiles": "C1=CC=C(C=C1)/C=C/C[NH3+]",  "name": "Cinnamylammonium"},

    # === Additional known spacer cations from literature ===
    "3F-PEA2": {"smiles": "NCCC1=CC=C(F)C=C1",  "name": "3-Fluorophenethylammonium (alt name)"},
    "4CF3-PEA": {"smiles": "NCCC1=CC=C(C(F)(F)F)C=C1","name": "4-(Trifluoromethyl)phenethylammonium"},
    "4SCN-PEA": {"smiles": "NCCC1=CC=C(SC#N)C=C1","name": "4-Thiocyanatophenethylammonium"},

    # === Pyridinium ethyl ammoniums (from RSC review) ===
    "PyREA":  {"smiles": "C1=CC=[NH+]C=C1CC[NH3+]","name": "4-(2-Ammonioethyl)pyridinium"},
    "2PyREA": {"smiles": "C1=CC=C[NH+]=C1CC[NH3+]","name": "2-(2-Ammonioethyl)pyridinium"},
    "3PyREA": {"smiles": "C1=C[NH+]=CC=C1CC[NH3+]","name": "3-(2-Ammonioethyl)pyridinium"},
    "4PyREA": {"smiles": "C1=C[NH+]=CC(=C1)CC[NH3+]","name": "4-(2-Ammonioethyl)pyridinium (isomer)"},

    # === Triammonium / guanidinium-like triammoniums (charge +3) ===
    # Triammoniums with three -NH3+ groups, guanidinium-analog spacer cations

    # Linear aliphatic triammoniums
    "TAPA": {"smiles": "C(C[NH3+])(C[NH3+])C[NH3+]", "name": "Tris(2-ammonioethyl)amine"},  # TREN-like, charge +3
    "DPTA": {"smiles": "C(C[NH3+])C[NH3+]",            "name": "Dipropylenetriammonium"},      # Charge +2
    "SPMD": {"smiles": "C(CCC[NH3+])CC[NH3+]",         "name": "Spermidinium"},                # Charge +2
    "SPMN": {"smiles": "C(CCC[NH3+])NC(CC[NH3+])C[NH3+]","name": "Sperminium"},               # Charge +3
    "TREN": {"smiles": "C(C[NH3+])N(CC[NH3+])CC[NH3+]", "name": "Tris(2-ammonioethyl)ammonium"},  # Charge +3
    "TAPA3":{"smiles": "C(CC[NH3+])C([NH3+])C[NH3+]",  "name": "1,3,5-Triammoniopentane"},
    "TAHA": {"smiles": "C(CCC[NH3+])C([NH3+])CCC[NH3+]","name": "1,3,5-Triammoniohexane"},

    # Aromatic triammoniums
    "TAB":  {"smiles": "C1=C(C=C(C=C1[NH3+])[NH3+])[NH3+]",      "name": "Benzene-1,3,5-triammonium"},
    "TABMe":{"smiles": "CC1=CC(=CC(=C1[NH3+])[NH3+])[NH3+]",     "name": "2,4,6-Trimethylbenzene-1,3,5-triammonium"},
    "TABF": {"smiles": "C1=C(C(=C(C(=C1F)[NH3+])[NH3+])F)[NH3+]","name": "2,4,6-Trifluorobenzene-1,3,5-triammonium"},
    "TMB":  {"smiles": "C1(=CC(=CC(=C1)C[NH3+])C[NH3+])C[NH3+]","name": "1,3,5-Tris(ammoniomethyl)benzene"},
    "TEB":  {"smiles": "C1(=CC(=CC(=C1)CC[NH3+])CC[NH3+])CC[NH3+]","name": "1,3,5-Tris(2-ammonioethyl)benzene"},

    # Guanidinium-derived triammoniums
    "TGUA": {"smiles": "NC(=[NH2+])NNC(=[NH2+])N",  "name": "Triaminoguanidinium"},        # TAG
    "BIGUA":{"smiles": "NC(=[NH2+])NC(=[NH2+])N",    "name": "Biguanidinium"},              # Biguanide
    "DABCO":{"smiles": "C1CN2CC[NH+]1CC2",           "name": "DABCO monoammonium"},         # DABCOH+, charge +1
    "TMGUA":{"smiles": "CN(C)C(=[NH2+])N(C)C",        "name": "1,1,3,3-Tetramethylguanidinium"},  # TMG, charge +1

    # Cyclic / cage triammoniums
    "CYTA": {"smiles": "C1C(CC(CC1[NH3+])[NH3+])[NH3+]",   "name": "cis,cis-1,3,5-Triammoniocyclohexane"},
    "CYTT": {"smiles": "C1C(CC(CC1[NH3+])[NH3+])[NH3+]",   "name": "trans,trans-1,3,5-Triammoniocyclohexane"},
    "ADTA": {"smiles": "C1C2CC3CC1CC(C2)(C3)C([NH3+])([NH3+])[NH3+]","name": "1,3,5-Triammonioadamantane"},
    "TACN": {"smiles": "C1CN(CC[NH2+]1)CC[NH3+]",           "name": "1,4,7-Triazacyclononane diammonium"},  # TACN, charge +2

    # Tripodal triammoniums (special class for 2D HOIPs)
    "TPMA": {"smiles": "C1=CC=C(C=C1)C(C[NH3+])(C[NH3+])C[NH3+]","name": "Tris(ammoniomethyl)phenylmethane"},
    "TPEA": {"smiles": "C1=CC=C(C=C1)C(CC[NH3+])(CC[NH3+])CC[NH3+]","name": "Tris(2-ammonioethyl)phenylmethane"},
    "TTMA": {"smiles": "C1=CC(=CC=C1)C(C2=CC=C(C=C2)[NH3+])(C3=CC=C(C=C3)[NH3+])[NH3+]","name": "Tris(4-ammoniophenyl)methane"},

    # Oligoamine triammoniums
    "TETA": {"smiles": "C(C[NH3+])[NH2+]CC[NH2+]CC[NH3+]",    "name": "Triethylenetetraammonium"},   # TETA, charge +4
    "PEHA": {"smiles": "C(CN)CNCCNCCNCCN",           "name": "Pentaethylenehexaammonium"},  # PEHA, charge +6 (neutral SMILES for RDKit compatibility)

    # Guanidinium-substituted ammoniums (hybrid charge +1 but guanidine-like)
    "GABA": {"smiles": "N=C(N)NCC[NH3+]",        "name": "2-Guanidinoethylammonium"},
    "GAPA": {"smiles": "N=C(N)NCCC[NH3+]",       "name": "3-Guanidinopropylammonium"},
    "GABUT":{"smiles": "N=C(N)NCCCC[NH3+]",      "name": "4-Guanidinobutylammonium"},
    "GAPEN":{"smiles": "N=C(N)NCCCCC[NH3+]",     "name": "5-Guanidinopentylammonium"},
    "GAHEX":{"smiles": "N=C(N)NCCCCCC[NH3+]",    "name": "6-Guanidinohexylammonium"},
    "AGMAT":{"smiles": "N=C(N)NCCCC[NH3+]",      "name": "Agmatine (4-guanidinobutylammonium)"},

    # Amino acid-derived ammoniums (+1 charge, zwitterionic character)
    "ORN":  {"smiles": "NCCCC([NH3+])C(=O)O",    "name": "Ornithinium"},   # Ornithine protonated
    "LYS":  {"smiles": "NCCCCC([NH3+])C(=O)O",   "name": "Lysinium"},      # Lysine protonated
    "ARG":  {"smiles": "NC(=[NH2+])NCCCC([NH3+])C(=O)O","name": "Argininium"},  # Arginine protonated
    "HIS":  {"smiles": "C1=C(NC=N1)CC([NH3+])C(=O)O","name": "Histidinium"},  # Histidine protonated
}

# ===========================================================================
# ASSEMBLE FULL LIBRARY
# ===========================================================================

def get_full_cation_library():
    """Returns combined dictionary of all cations with role classification."""
    library = {}

    for abbr, info in A_SITE_CATIONS.items():
        library[abbr] = {
            "smiles": info["smiles"],
            "charge": 1,
            "role": "A-site cation",
            "name": info["name"],
        }

    for abbr, info in SPACER_CATIONS.items():
        # Detect charge from ammonium group count in SMILES
        # Count [NH3+] occurrences in the SMILES string as primary charge indicator
        smi = info["smiles"]
        n_ammonio = smi.count("[NH3+]")
        n_nh2plus = smi.count("[NH2+]")
        n_nhplus = smi.count("[NH+]")
        n_nplus = smi.count("[N+]")
        
        # Base charge from terminal ammonium groups
        base_charge = n_ammonio
        
        # For guanidinium-like groups [NH2+]=C-N: each adds +1
        # For pyridinium/imidazolium-like [NH+]: adds +1
        # For quaternary ammonium [N+](C)(C)C: adds +1
        base_charge += n_nh2plus + n_nhplus + n_nplus
        
        # Clamp to reasonable range
        if base_charge < 1:
            # For neutral SMILES (like PEHA), estimate from nitrogen count
            n_n = smi.count("N")
            if n_n > 1 and "hexaammonium" in info["name"].lower():
                base_charge = 6
            elif n_n > 1 and "tetraammonium" in info["name"].lower():
                base_charge = 4
            else:
                base_charge = 1
        if base_charge > 6:
            base_charge = 6
        
        charge = base_charge
        library[abbr] = {
            "smiles": info["smiles"],
            "charge": charge,
            "role": "spacer cation",
            "name": info["name"],
        }

    return library


def get_cation_counts():
    """Returns counts by category."""
    return {
        "A-site cations (small, 3D)": len(A_SITE_CATIONS),
        "Spacer cations (large, 2D)": len(SPACER_CATIONS),
        "Total": len(A_SITE_CATIONS) + len(SPACER_CATIONS),
    }


if __name__ == "__main__":
    lib = get_full_cation_library()
    counts = get_cation_counts()
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"\n  First 5 A-site: {list(A_SITE_CATIONS.keys())[:5]}")
    print(f"  First 5 spacer: {list(SPACER_CATIONS.keys())[:5]}")
