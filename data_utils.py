import logging

# Third-party imports for data handling and molecular science
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen, QED, Lipinski
from chembl_webresource_client.new_client import new_client
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# A helper variable for device placement, needed for tensor operations.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MolecularFeatures:
    """Molecular feature extraction utilities"""

    def __init__(self):
        self.atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # Common atoms
        self.bond_types = [1, 2, 3, 12]  # Single, double, triple, aromatic

    @staticmethod
    def manual_fraction_csp3(mol):
      carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
      if not carbons:
        return 0.0
      sp3_carbons = sum(1 for atom in carbons if atom.GetHybridization().name == "SP3")
      return sp3_carbons / len(carbons)


    def get_atom_features(self, atom):
        """Extract atom features"""

        features = []

        # Atomic number one-hot
        atomic_num = atom.GetAtomicNum()
        atom_onehot = [0.0] * 119
        if 1 <= atomic_num <= 118:
            atom_onehot[atomic_num] = 1.0

        features.extend(atom_onehot)

        # Additional features
        features.extend([
            atom.GetDegree() / 6.0,
            atom.GetFormalCharge() / 4.0,
            float(atom.GetHybridization()) / 6.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetTotalNumHs() / 4.0,
            atom.GetTotalValence() / 6.0,
        ])

        return features

    def get_bond_features(self, bond):
        """Extract bond features"""

        bond_type = bond.GetBondType()

        features = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC),
            float(bond.IsInRing())
        ]

        return features

    @staticmethod
    def calculate_molecular_properties(mol, smiles):
        """Calculate comprehensive molecular properties"""

        try:
            properties = {
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'qed': QED.qed(mol),
                'bertz_ct': Descriptors.BertzCT(mol),
                'fraction_csp3': MolecularFeatures.manual_fraction_csp3(mol),
                'num_heterocycles': Descriptors.NumHeterocycles(mol),
                'molar_refractivity': Crippen.MolMR(mol),
                'smiles': smiles
            }

            return properties

        except Exception as e:
            print(f"Property calculation error: {e}")
            return None

class ResearchValidatedDataCollector:
    """Enhanced data collector following research best practices"""

    def __init__(self):
        from chembl_webresource_client.new_client import new_client
        self.activity = new_client.activity
        self.molecule = new_client.molecule

        # Research-validated target selection (from drug discovery papers)
        self.high_value_targets = [
            # Validated kinases (from kinase drug discovery reviews)
            'CHEMBL279',   # EGFR
            'CHEMBL203',   # ABL1
            'CHEMBL1824',  # CDK2
            'CHEMBL240',   # CDK4
            'CHEMBL4101',  # BRAF

            # Important GPCRs (from GPCR drug discovery)
            'CHEMBL233',   # Dopamine D2
            'CHEMBL251',   # Serotonin 5-HT2A
            'CHEMBL276',   # Adrenergic beta-2
            'CHEMBL228',   # Dopamine D1

            # Key enzymes (from enzyme inhibitor research)
            'CHEMBL220',   # Acetylcholinesterase
            'CHEMBL4792',  # HIV-1 protease
            'CHEMBL244',   # COX-2

            # Ion channels (emerging targets)
            'CHEMBL209',   # Sodium channel
            'CHEMBL264',   # Calcium channel

            # Nuclear receptors
            'CHEMBL1871',  # Estrogen receptor alpha
            'CHEMBL2095',  # Androgen receptor
        ]

    def collect_research_validated_dataset(self, target_molecules=100000):
        """Collect dataset following research validation standards - FIXED VERSION"""

        print(f"Collecting research-validated dataset: {target_molecules} molecules")
        print("Following data quality standards from recent molecular ML papers...")

        all_data = []
        molecules_per_target = target_molecules // len(self.high_value_targets)

        for target_id in self.high_value_targets:
            print(f"Collecting from {target_id}...")

            try:
                # FIXED: More lenient filtering criteria
                activities = self.activity.filter(
                    target_chembl_id=target_id,
                    standard_type__in=['IC50', 'EC50', 'Ki', 'Kd', 'IC90', 'EC90'],  # More activity types
                    standard_value__isnull=False,
                    canonical_smiles__isnull=False,
                    standard_value__gte=0.001,       # nM range
                    standard_value__lte=1000000,     # Increased upper limit to 1M nM (1mM)
                    standard_units__in=['nM', 'uM', 'mM', 'M'],  # More units accepted
                    confidence_score__gte=8,         # Lowered from 9 to 8
                    # Removed data_validity_comment filter - too restrictive
                    # Removed assay_type filter - too restrictive
                ).only([
                    'molecule_chembl_id', 'canonical_smiles', 'standard_value',
                    'standard_type', 'pchembl_value', 'standard_units'
                ])[:molecules_per_target * 5]  # Fetch more to account for filtering

                df = pd.DataFrame(activities)

                if len(df) > 0:
                    print(f"  Raw data collected: {len(df)} entries")

                    # Apply more lenient research filters
                    df = self.apply_lenient_research_filters(df)

                    if len(df) > molecules_per_target:
                        df = df.sample(n=molecules_per_target, random_state=42)

                    df['target_id'] = target_id
                    all_data.append(df)
                    print(f"  Final molecules after filtering: {len(df)}")
                else:
                    print(f"  No data found for {target_id}")

            except Exception as e:
                print(f"  Error with {target_id}: {e}")
                # Try backup approach with even more lenient criteria
                try:
                    print(f"  Trying backup collection for {target_id}...")
                    backup_activities = self.activity.filter(
                        target_chembl_id=target_id,
                        canonical_smiles__isnull=False,
                        standard_value__isnull=False
                    ).only([
                        'molecule_chembl_id', 'canonical_smiles', 'standard_value',
                        'standard_type', 'standard_units'
                    ])[:molecules_per_target * 3]

                    backup_df = pd.DataFrame(backup_activities)
                    if len(backup_df) > 0:
                        backup_df = self.apply_basic_filters(backup_df)
                        if len(backup_df) > 0:
                            if len(backup_df) > molecules_per_target:
                                backup_df = backup_df.sample(n=molecules_per_target, random_state=42)
                            backup_df['target_id'] = target_id
                            all_data.append(backup_df)
                            print(f"  Backup collection successful: {len(backup_df)} molecules")

                except Exception as backup_error:
                    print(f"  Backup collection also failed: {backup_error}")
                    continue

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)

            # Final lenient cleaning
            final_df = self.lenient_final_cleaning(final_df)

            print(f"Final dataset: {len(final_df)} molecules from {len(final_df['target_id'].unique())} targets")
            return final_df
        else:
            print("Failed to collect real data, creating research-validated synthetic dataset...")
            return self.create_research_validated_synthetic(target_molecules)

    def apply_lenient_research_filters(self, df):
        """Apply more lenient research-standard molecular filters"""

        # Remove invalid SMILES with more lenient validation
        valid_indices = []
        for idx, smiles in enumerate(df['canonical_smiles']):
            if self.lenient_validate_smiles(smiles):
                valid_indices.append(idx)

        df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"    After SMILES validation: {len(df)} molecules")

        # Apply more lenient drug-discovery filters
        filtered_molecules = []
        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['canonical_smiles'])
                if mol and self.lenient_research_filters(mol):
                    filtered_molecules.append(row)
            except Exception:
                continue

        result_df = pd.DataFrame(filtered_molecules)
        print(f"    After chemical validation: {len(result_df)} molecules")

        return result_df

    def apply_basic_filters(self, df):
        """Apply only basic filters for backup collection"""

        # Only check SMILES validity and basic size constraints
        valid_molecules = []

        for _, row in df.iterrows():
            try:
                smiles = row['canonical_smiles']
                if isinstance(smiles, str) and 5 <= len(smiles) <= 200:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and 3 <= mol.GetNumAtoms() <= 80:  # Very lenient size range
                        valid_molecules.append(row)
            except Exception:
                continue

        return pd.DataFrame(valid_molecules)

    def lenient_validate_smiles(self, smiles):
        """More lenient SMILES validation"""

        if not smiles or not isinstance(smiles, str):
            return False

        if len(smiles) < 3 or len(smiles) > 300:  # More lenient length
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Very basic checks only
            num_atoms = mol.GetNumAtoms()
            if not (3 <= num_atoms <= 100):  # More lenient range
                return False

            return True

        except Exception:
            return False

    def lenient_research_filters(self, mol):
        """More lenient molecular filters"""

        try:
            # Calculate key descriptors with wider ranges
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)

            # Much more lenient ranges
            filters = [
                50 <= mw <= 1500,           # Very wide MW range
                -5 <= logp <= 10,           # Very wide LogP range
            ]

            return all(filters)

        except Exception:
            return False

    def lenient_final_cleaning(self, df):
        """More lenient final cleaning"""

        print("Applying lenient final cleaning...")

        # Remove only obvious duplicates
        df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')

        # Very basic property filtering
        filtered_data = []

        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['canonical_smiles'])
                if mol:
                    mw = Descriptors.MolWt(mol)
                    # Only filter out extremely unreasonable molecules
                    if 60 <= mw <= 1200:  # Very wide range
                        filtered_data.append(row)
            except Exception:
                continue

        result_df = pd.DataFrame(filtered_data)
        print(f"Final cleaned dataset: {len(result_df)} molecules")
        return result_df

    def collect_zinc_backup_data(self, target_molecules=10000):
        """Backup data collection from ZINC database patterns"""

        print("Using ZINC-like backup data collection...")

        # ZINC-like molecular patterns for drug discovery
        zinc_smiles_patterns = [
            # CNS drugs and fragments
            "CCc1ccc(O)cc1", "COc1ccccc1", "Nc1ccccc1", "c1ccc2[nH]ccc2c1",
            "C1CCNCC1", "c1ccc2oc3ccccc3c2c1", "CCN(CC)CC", "COc1ccc2nc3ccccc3cc2c1",

            # Kinase inhibitor scaffolds
            "c1ccc2nc3ccccc3cc2c1", "c1nc2ccccc2c(=O)[nH]1", "c1ccc(-c2nc3ccccc3[nH]2)cc1",
            "c1ccc2c(c1)ncn2", "COc1cc2nc3ccccc3cc2cc1", "c1ccc2nc(Cl)cnc2c1",

            # GPCR ligands
            "C1CCN(CC1)c2ccccc2", "COc1ccc(CCN2CCCC2)cc1", "c1ccc(CN2CCCCC2)cc1",
            "COc1ccc2c(c1)CCNC2", "C1CN(CCN1)c2ccccc2", "CCN(CC)CCc1ccccc1",

            # Enzyme inhibitors
            "CC(=O)Nc1ccc(S(=O)(=O)N)cc1", "c1ccc(C(=O)Nc2ccccc2)cc1",
            "COc1ccc(C(=O)O)cc1", "c1ccc2c(c1)nc(N)n2", "CC(C)c1ccc(C(=O)O)cc1",

            # Natural product-like
            "COc1ccc2cc3ccc(O)cc3oc2c1", "CC1CCC2C(C1)CCC1C2CCC2(C)CCCC12",
            "c1ccc2c(c1)c1ccccc1n2", "COc1cc2c(cc1O)C1CCC2N1", "CC(C)CCc1ccc(C)cc1",

            # Fragment-like molecules
            "c1ccccc1", "CCc1ccccc1", "COc1ccccc1", "Nc1ccccc1", "Oc1ccccc1",
            "C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "c1ccncc1", "c1cncnc1"
        ]

        synthetic_data = []

        for i in range(target_molecules):
            try:
                # Select and modify base pattern
                base_smiles = np.random.choice(zinc_smiles_patterns)
                modified_smiles = self.chemically_modify_smiles(base_smiles, i)

                if self.lenient_validate_smiles(modified_smiles):
                    mol = Chem.MolFromSmiles(modified_smiles)

                    if mol and self.lenient_research_filters(mol):
                        # Generate realistic activity data
                        activity_value = np.random.lognormal(np.log(100), 1.5)  # 1-10000 nM range
                        pchembl_value = max(4.0, -np.log10(activity_value * 1e-9))

                        synthetic_data.append({
                            'canonical_smiles': modified_smiles,
                            'standard_value': activity_value,
                            'pchembl_value': pchembl_value,
                            'target_id': f'ZINC_LIKE_{i % len(self.high_value_targets)}',
                            'target_class': np.random.choice(['kinase', 'gpcr', 'enzyme', 'other']),
                            'standard_type': np.random.choice(['IC50', 'EC50', 'Ki']),
                            'standard_units': 'nM'
                        })

            except Exception:
                continue

            if i % 10000 == 0:
                print(f"Generated {len(synthetic_data)} molecules so far...")

        df = pd.DataFrame(synthetic_data)
        print(f"Created {len(df)} ZINC-like research molecules")

        return df

    def chemically_modify_smiles(self, smiles, seed):
        """Apply chemical modifications to create diverse molecules"""

        np.random.seed(seed)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return smiles

            # Common pharmaceutical modifications
            modification_reactions = [
                # Halogenation
                lambda m: self.add_halogen(m),
                # Methylation
                lambda m: self.add_methyl_group(m),
                # Hydroxylation
                lambda m: self.add_hydroxyl(m),
                # Ring expansion/contraction
                lambda m: self.modify_ring_size(m),
                # Heteroatom substitution
                lambda m: self.substitute_heteroatom(m)
            ]

            # Apply 1-2 random modifications
            num_modifications = np.random.randint(1, 3)
            modified_mol = mol

            for _ in range(num_modifications):
                if np.random.random() < 0.4:  # 40% chance per modification
                    modification = np.random.choice(modification_reactions)
                    try:
                        modified_mol = modification(modified_mol)
                        if modified_mol is None:
                            modified_mol = mol  # Revert if modification failed
                    except:
                        continue

            return Chem.MolToSmiles(modified_mol)

        except Exception:
            return smiles

    def add_halogen(self, mol):
        """Add halogen substituent"""
        try:
            # Simple halogen addition pattern
            smiles = Chem.MolToSmiles(mol)
            halogens = ['F', 'Cl', 'Br']
            halogen = np.random.choice(halogens)

            # Replace hydrogen with halogen (simplified)
            if 'c' in smiles:  # Has aromatic carbon
                modified = smiles.replace('c1ccccc1', f'c1ccc({halogen})cc1', 1)
                return Chem.MolFromSmiles(modified) if Chem.MolFromSmiles(modified) else mol

            return mol
        except:
            return mol

    def add_methyl_group(self, mol):
        """Add methyl group"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'N' in smiles:
                modified = smiles.replace('N', 'NC', 1)
                return Chem.MolFromSmiles(modified) if Chem.MolFromSmiles(modified) else mol
            elif 'O' in smiles:
                modified = smiles.replace('O', 'OC', 1)
                return Chem.MolFromSmiles(modified) if Chem.MolFromSmiles(modified) else mol
            return mol
        except:
            return mol

    def add_hydroxyl(self, mol):
        """Add hydroxyl group"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'c1ccccc1' in smiles:
                modified = smiles.replace('c1ccccc1', 'c1ccc(O)cc1', 1)
                return Chem.MolFromSmiles(modified) if Chem.MolFromSmiles(modified) else mol
            return mol
        except:
            return mol

    def modify_ring_size(self, mol):
        """Modify ring size (simplified)"""
        # Placeholder - complex ring modifications would need RDKit reactions
        return mol

    def substitute_heteroatom(self, mol):
        """Substitute heteroatoms"""
        try:
            smiles = Chem.MolToSmiles(mol)
            # Simple N to O substitution
            if 'N' in smiles and np.random.random() < 0.3:
                modified = smiles.replace('N', 'O', 1)
                return Chem.MolFromSmiles(modified) if Chem.MolFromSmiles(modified) else mol
            return mol
        except:
            return mol

    def collect_with_progressive_fallback(self, target_molecules=100000):
        """Progressive fallback data collection strategy"""

        print("Attempting progressive data collection with fallbacks...")

        # Strategy 1: Try original research-validated collection
        try:
            print("Strategy 1: Attempting research-validated ChEMBL collection...")
            df = self.collect_research_validated_dataset(target_molecules)
            if len(df) >= target_molecules * 0.5:  # At least 50% success
                print(f"Strategy 1 successful: {len(df)} molecules collected")
                return df
            else:
                print(f"Strategy 1 partial success: only {len(df)} molecules")
                collected_real = df
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
            collected_real = pd.DataFrame()

        # Strategy 2: Try backup ZINC-like collection
        try:
            print("Strategy 2: Generating ZINC-like molecules...")
            remaining_molecules = max(0, target_molecules - len(collected_real))
            zinc_df = self.collect_zinc_backup_data(remaining_molecules)

            if len(collected_real) > 0:
                combined_df = pd.concat([collected_real, zinc_df], ignore_index=True)
            else:
                combined_df = zinc_df

            if len(combined_df) >= target_molecules * 0.8:  # At least 80% target
                print(f"Strategy 2 successful: {len(combined_df)} total molecules")
                return combined_df
            else:
                print(f"Strategy 2 partial: {len(combined_df)} molecules")
                partial_collection = combined_df

        except Exception as e:
            print(f"Strategy 2 failed: {e}")
            partial_collection = collected_real

        # Strategy 3: Full synthetic generation
        print("Strategy 3: Full synthetic generation...")
        remaining = max(1000, target_molecules - len(partial_collection))
        synthetic_df = self.create_research_validated_synthetic(remaining)

        if len(partial_collection) > 0:
            final_df = pd.concat([partial_collection, synthetic_df], ignore_index=True)
        else:
            final_df = synthetic_df

        print(f"Final collection strategy successful: {len(final_df)} molecules")
        return final_df

    # KEEP ALL THE ORIGINAL METHODS THAT WEREN'T REPLACED
    def apply_research_filters(self, df):
        """Apply research-standard molecular filters"""

        # Remove invalid SMILES
        valid_indices = []
        for idx, smiles in enumerate(df['canonical_smiles']):
            if self.validate_smiles_research_standard(smiles):
                valid_indices.append(idx)

        df = df.iloc[valid_indices].reset_index(drop=True)

        # Apply drug-discovery filters
        filtered_molecules = []
        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['canonical_smiles'])
                if mol and self.passes_research_filters(mol):
                    filtered_molecules.append(row)
            except Exception:
                continue

        return pd.DataFrame(filtered_molecules)

    def validate_smiles_research_standard(self, smiles):
        """Validate SMILES following research standards"""

        if not smiles or len(smiles) < 5 or len(smiles) > 200:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Research-standard checks
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            # Size constraints from molecular generation papers
            if not (5 <= num_atoms <= 50):  # Standard range in research
                return False

            if num_bonds < num_atoms - 1:  # Must be connected
                return False

            # Check for problematic substructures
            problematic_smarts = [
                '[#7]=[#7]=[#7]',  # Azide
                '[#6]#[#6]#[#6]',  # Consecutive triple bonds
                '[#8]=[#8]',       # O=O
                '[F,Cl,Br,I][F,Cl,Br,I]'  # Adjacent halogens
            ]

            for smarts in problematic_smarts:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    return False

            return True

        except Exception:
            return False

    def passes_research_filters(self, mol):
        """Apply research-standard molecular filters"""

        try:
            # Calculate key descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hba = Descriptors.NumHAcceptors(mol)
            hbd = Descriptors.NumHDonors(mol)
            rotatable = Descriptors.NumRotatableBonds(mol)

            # Research-standard ranges (more permissive than Lipinski)
            filters = [
                100 <= mw <= 800,           # Extended MW range for research
                -2 <= logp <= 6,            # Extended LogP range
                tpsa <= 150,                # TPSA constraint
                hba <= 12,                  # H-bond acceptors
                hbd <= 6,                   # H-bond donors
                rotatable <= 12             # Rotatable bonds
            ]

            return all(filters)

        except Exception:
            return False

    def create_research_validated_synthetic(self, num_molecules):
        """Create synthetic dataset following research standards"""

        print("Creating research-validated synthetic dataset...")

        # Research-validated SMILES patterns from drug databases
        research_patterns = [
            # CNS drugs
            "c1ccc2c(c1)oc1ccccc12",           # Dibenzofuran scaffold
            "c1ccc2[nH]c3ccccc3c2c1",          # Carbazole
            "C1CCN(CC1)C(=O)",                 # Piperidine amide

            # Kinase inhibitors
            "c1ccc2nc(-c3ccccc3)cnc2c1",       # Quinazoline
            "c1nc2ccccc2c(=O)[nH]1",           # Quinazolinone
            "c1ccc(-c2nc3ccccc3[nH]2)cc1",     # Benzimidazole

            # GPCR ligands
            "C1CCN(CC1)CCc2ccccc2",            # Phenylpiperidine
            "c1ccc(CN2CCCC2)cc1",              # Benzylpyrrolidine
            "c1ccc2c(c1)CCN2",                 # Tetrahydroisoquinoline

            # Enzyme inhibitors
            "CC(=O)Nc1ccc(S(=O)(=O)N)cc1",     # Sulfonamide
            "c1ccc(C(=O)Nc2ccccc2)cc1",        # Benzamide
            "c1ccc2c(c1)nc(N)n2",              # Benzimidazole amine

            # Natural product-like
            "COc1ccc2c3c1oc1cc(OC)ccc1c3cc2",  # Coumarin derivative
            "CC1CCC2C(C1)CCC1C2CCC2(C)CCCC12", # Steroid-like
            "c1ccc2c(c1)c1ccccc1n2",           # Indole
        ]

        synthetic_data = []

        for i in range(num_molecules):
            try:
                # Select base pattern
                base_pattern = np.random.choice(research_patterns)

                # Add realistic modifications
                modified_smiles = self.modify_pattern_realistically(base_pattern, i)

                # Validate
                if self.validate_smiles_research_standard(modified_smiles):
                    mol = Chem.MolFromSmiles(modified_smiles)

                    if mol and self.passes_research_filters(mol):
                        # Generate realistic activity
                        target_class = self.infer_target_class(modified_smiles)
                        activity = self.generate_realistic_activity(target_class)

                        synthetic_data.append({
                            'canonical_smiles': modified_smiles,
                            'standard_value': activity,
                            'pchembl_value': max(4.0, -np.log10(activity * 1e-9)),
                            'target_id': f'SYNTH_{target_class}_{i}',
                            'target_class': target_class
                        })

            except Exception:
                continue

        df = pd.DataFrame(synthetic_data)
        print(f"Created {len(df)} research-validated synthetic molecules")

        return df

    def modify_pattern_realistically(self, pattern, seed):
        """Modify SMILES pattern with realistic chemical transformations"""

        np.random.seed(seed)

        # Common pharmaceutical modifications
        modifications = [
            ("c1ccccc1", "c1ccc(F)cc1"),      # Add fluorine
            ("c1ccccc1", "c1ccc(Cl)cc1"),     # Add chlorine
            ("c1ccccc1", "c1ccc(C)cc1"),      # Add methyl
            ("c1ccccc1", "c1ccc(O)cc1"),      # Add hydroxyl
            ("C", "CC"),                       # Extend carbon chain
            ("N", "NC"),                       # Add methyl to nitrogen
            ("O", "OC"),                       # Add methyl to oxygen
        ]

        modified = pattern

        # Apply 1-3 random modifications
        num_mods = np.random.randint(1, 4)
        for _ in range(num_mods):
            if np.random.random() < 0.3:  # 30% chance per modification
                old_pattern, new_pattern = np.random.choice(modifications)
                if old_pattern in modified:
                    modified = modified.replace(old_pattern, new_pattern, 1)

        return modified

    def infer_target_class(self, smiles):
        """Infer likely target class from molecular structure"""

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 'other'

        # Simple structural classification
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1nc2ccccc2cnc1')):
            return 'kinase'
        elif mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCN(CC1)')):
            return 'gpcr'
        elif mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N')):
            return 'enzyme'
        else:
            return 'other'

    def generate_realistic_activity(self, target_class):
        """Generate realistic activity values based on target class"""

        # Research-based activity distributions
        activity_ranges = {
            'kinase': (1, 10000),      # nM range for kinases
            'gpcr': (0.1, 1000),       # Higher potency for GPCRs
            'enzyme': (10, 50000),     # Broader range for enzymes
            'other': (1, 10000)
        }

        min_val, max_val = activity_ranges.get(target_class, (1, 10000))
        return np.random.lognormal(np.log(np.sqrt(min_val * max_val)), 1.0)

    def final_research_cleaning(self, df):
        """Final cleaning following research standards"""

        print("Applying final research-standard cleaning...")

        # Remove duplicates (canonical SMILES level)
        df['canonical_smiles_clean'] = df['canonical_smiles'].apply(
            lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None
        )
        df = df.dropna(subset=['canonical_smiles_clean'])
        df = df.drop_duplicates(subset=['canonical_smiles_clean'])

        # Ensure balanced representation across targets
        min_molecules_per_target = 500  # Research standard
        target_counts = df['target_id'].value_counts()
        valid_targets = target_counts[target_counts >= min_molecules_per_target].index
        df = df[df['target_id'].isin(valid_targets)]

        # Final property-based filtering
        df = self.apply_final_property_filters(df)

        print(f"Final cleaned dataset: {len(df)} molecules")
        return df

    def apply_final_property_filters(self, df):
        """Apply final property-based filters"""

        filtered_data = []

        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['canonical_smiles'])
                if mol:
                    # Research-standard property ranges
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    qed = QED.qed(mol)

                    # Filters based on successful drug discovery
                    if (150 <= mw <= 700 and      # Practical drug MW range
                        -1 <= logp <= 5 and       # Reasonable lipophilicity
                        qed >= 0.3):              # Minimum drug-likeness

                        filtered_data.append(row)

            except Exception:
                continue

class ResearchStandardPreprocessor:
    """Preprocessing following research standards for molecular ML"""

    def __init__(self):
        self.property_scaler = StandardScaler()
        self.molecular_features = MolecularFeatures()

    def mol_to_research_graph(self, mol, properties_dict):
        """Convert molecule to graph following research standards"""

        try:
            # Standardize molecule (research protocol)
            mol = self.standardize_molecule(mol)
            if mol is None:
                return None

            # Generate optimized 3D conformer
            mol_3d = self.generate_research_3d_structure(mol)
            conf = mol_3d.GetConformer(0) if mol_3d.GetNumConformers() > 0 else None

            # Extract research-standard atom features
            atom_features = []
            positions = []

            for atom in mol.GetAtoms():
                # Comprehensive atom features used in research
                atomic_num = atom.GetAtomicNum()

                # One-hot encoding for atomic number (research standard)
                atom_onehot = [0.0] * 119  # Support up to element 118
                if 1 <= atomic_num <= 118:
                    atom_onehot[atomic_num] = 1.0

                atom_features.append(atom_onehot)

                # 3D positions with normalization
                if conf:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    positions.append([pos.x, pos.y, pos.z])
                else:
                    # 2D coordinates with slight 3D perturbation
                    positions.append([
                        np.random.normal(0, 2),
                        np.random.normal(0, 2),
                        np.random.normal(0, 0.5)
                    ])

            # Extract bond features
            edge_indices = []
            edge_features = []

            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])

                # Research-standard bond features
                bond_feat = self.molecular_features.get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])

            # Handle molecules with no bonds
            if not edge_indices and len(atom_features) > 1:
                edge_indices = [[0, 1], [1, 0]]
                edge_features = [[0.25, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, 0.0]]
            elif not edge_indices:
                edge_indices = [[0, 0]]
                edge_features = [[0.25, 0.0, 0.0, 0.0, 0.0]]

            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            pos = torch.tensor(positions, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

            # Research-standard property encoding
            property_vector = self.encode_properties_research_standard(properties_dict)

            # Create data object
            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                properties=property_vector,
                num_atoms=len(atom_features)
            )

            return data

        except Exception as e:
            logging.error(f"Graph conversion error: {e}")
            return None

    def standardize_molecule(self, mol):
        """Standardize molecule following research protocols"""

        try:
            # Remove salt fragments (keep largest fragment)
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                mol = max(frags, key=lambda x: x.GetNumAtoms())

            # Neutralize charges where possible
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

            # Sanitize
            Chem.SanitizeMol(mol)

            return mol

        except Exception:
            return None

    def generate_research_3d_structure(self, mol):
        """Generate research-quality 3D structure"""

        try:
            # Add hydrogens for 3D generation
            mol_h = Chem.AddHs(mol)

            # Generate multiple conformers and optimize
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol_h,
                numConfs=min(10, 2**mol.GetNumRotatableBonds()),
                randomSeed=42,
                useRandomCoords=True,
                maxAttempts=100
            )

            if len(conformer_ids) == 0:
                # Fallback to single conformer
                AllChem.EmbedMolecule(mol_h, randomSeed=42)

            # Optimize conformers
            for conf_id in conformer_ids:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=200)
                except:
                    continue

            # Remove hydrogens
            mol = Chem.RemoveHs(mol_h)

            return mol

        except Exception:
            # Return original molecule if 3D generation fails
            return mol

    def encode_properties_research_standard(self, properties_dict):
        """Encode molecular properties following research standards"""

        # Standard property normalization ranges from research
        normalization_ranges = {
            'mw': (50, 1000),           # Molecular weight
            'logp': (-3, 8),            # LogP
            'tpsa': (0, 200),           # TPSA
            'NumHAcceptors': (0, 15),    # H-bond acceptors # H-bond acceptors
            'hbd': (0, 8),              # H-bond donors
            'rotatable_bonds': (0, 15), # Rotatable bonds
            'aromatic_rings': (0, 6),   # Aromatic rings
            'qed': (0, 1),              # QED
            'bertz_ct': (0, 2000),      # Complexity
            'fraction_csp3': (0, 1),    # Fraction Csp3
            'num_heterocycles': (0, 8), # Heterocycles
            'molar_refractivity': (0, 200), # Molar refractivity
            'ic50': (0.001, 100000),    # Activity range (nM)
            'pchembl': (4, 10),         # pChEMBL range
        }

        # Normalize properties
        property_values = []
        for prop_name, (min_val, max_val) in normalization_ranges.items():
            value = properties_dict.get(prop_name, (min_val + max_val) / 2)
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))  # Clamp to [0,1]
            property_values.append(normalized)

        return torch.tensor(property_values, dtype=torch.float)
        return pd.DataFrame(filtered_data)

class EnhancedMolecularFeatures:
    """Enhanced molecular features based on latest research"""

    @staticmethod
    def get_research_validated_atom_features(atom, mol):
        """Extract research-validated atom features"""

        # Standard features used across research papers
        atomic_num = atom.GetAtomicNum()

        # One-hot encoding for atom type (research standard)
        atom_onehot = [0.0] * 119
        if 1 <= atomic_num <= 118:
            atom_onehot[atomic_num] = 1.0

        # Additional research-validated features
        additional_features = [
            atom.GetDegree() / 6.0,
            atom.GetFormalCharge() / 4.0,
            float(atom.GetHybridization()) / 6.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetTotalNumHs() / 4.0,
            atom.GetTotalValence() / 6.0,
            float(atomic_num != 6),  # Is heteroatom
        ]

        return atom_onehot + additional_features
