import copy
import importlib
from typing import Optional

import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from .reward_presets import get_task_preset
from .rl_diff import RL_Lightning
from ..util.rdkit import calc_energy, optimise_mol


class AdaptiveRL_Lightning(RL_Lightning):
    def __init__(
        self,
        gen: torch.nn.Module,
        vocab,
        lr: float,
        coord_scale: float = 1.0,
        use_ema: bool = True,
        compile_model: bool = True,
        warm_up_steps: Optional[int] = None,
        max_steps: int = 128,
        default_coord_noise_std: float = 0.2,
        default_cat_noise_level: float = 1.0,
        self_cond: bool = False,
        loss_weight: dict = {"types": 0.2, "bonds": 1.0, "charges": 1.0},
        formulation: str = "endpoint",
        eval_3D_props: bool = True,
        reward_name: str = "qed",
        reward_beta: float = 2.0,
        reward_weight_min: float = 0.1,
        reward_weight_max: float = 10.0,
        reward_norm_eps: float = 1e-6,
        anchor_weight: float = 0.1,
        anchor_loss_weight: float = 1.0,
        use_reference_anchor: bool = True,
        adaptive_time_sampling: bool = False,
        time_num_bins: int = 5,
        time_tau: float = 1.0,
        time_ema_beta: float = 0.1,
        time_prob_floor: float = 0.05,
        time_importance_correction: bool = True,
        time_importance_clip_min: float = 0.2,
        time_importance_clip_max: float = 5.0,
        reward_routing_enabled: bool = False,
        routed_loss_weights: Optional[dict] = None,
        constraints_enabled: bool = False,
        constraint_specs: Optional[dict] = None,
        dual_lr: float = 0.01,
        dual_max: float = 100.0,
        constraint_centering: bool = True,
        reward_groups: Optional[dict] = None,
        reward_transforms: Optional[dict] = None,
        reward_evaluators: Optional[dict] = None,
        task_preset_name: Optional[str] = None,
        group_coefficients: Optional[dict] = None,
        global_reference_smiles: Optional[str] = None,
        global_reference_mol: Optional[Chem.Mol] = None,
        required_smarts: Optional[list[str]] = None,
        forbidden_smarts: Optional[list[str]] = None,
        property_warning_once: bool = True,
    ):
        super().__init__(
            gen=gen,
            vocab=vocab,
            lr=lr,
            coord_scale=coord_scale,
            use_ema=use_ema,
            compile_model=compile_model,
            warm_up_steps=warm_up_steps,
            max_steps=max_steps,
            default_coord_noise_std=default_coord_noise_std,
            default_cat_noise_level=default_cat_noise_level,
            self_cond=self_cond,
            loss_weight=loss_weight,
            formulation=formulation,
            eval_3D_props=eval_3D_props,
            reward_name=reward_name,
            reward_beta=reward_beta,
            reward_weight_min=reward_weight_min,
            reward_weight_max=reward_weight_max,
            reward_norm_eps=reward_norm_eps,
            anchor_weight=anchor_weight,
            anchor_loss_weight=anchor_loss_weight,
            use_reference_anchor=use_reference_anchor,
        )

        self.adaptive_time_sampling = adaptive_time_sampling
        self.time_num_bins = max(1, int(time_num_bins))
        self.time_tau = max(float(time_tau), self.reward_norm_eps)
        self.time_ema_beta = float(time_ema_beta)
        self.time_prob_floor = float(time_prob_floor)
        self.time_importance_correction = time_importance_correction
        self.time_importance_clip_min = float(time_importance_clip_min)
        self.time_importance_clip_max = float(time_importance_clip_max)
        self.reward_routing_enabled = reward_routing_enabled
        self.constraints_enabled = constraints_enabled
        self.dual_lr = float(dual_lr)
        self.dual_max = float(dual_max)
        self.constraint_centering = constraint_centering
        self.reward_evaluators = reward_evaluators or {}
        self.task_preset_name = task_preset_name
        self.property_warning_once = property_warning_once
        self._warned_messages = set()
        self._optional_modules = {}
        self._posebusters = None
        self._init_optional_modules()
        self.required_smarts = required_smarts or []
        self.forbidden_smarts = forbidden_smarts or []
        self.reward_transforms = copy.deepcopy(reward_transforms or {})
        self._global_reference_mol = global_reference_mol
        if self._global_reference_mol is None and global_reference_smiles is not None:
            self._global_reference_mol = Chem.MolFromSmiles(global_reference_smiles)
        self._global_reference_fp = self._mol_to_fp(self._global_reference_mol) if self._global_reference_mol is not None else None

        preset = self._load_task_preset(task_preset_name) if task_preset_name is not None else {}
        default_groups = preset.get("reward_groups") or {"main": {"terms": {f"{reward_name}_reward": 1.0}, "coefficient": 1.0}}
        default_constraints = preset.get("constraint_specs") or {}
        default_routing = preset.get("routed_loss_weights") or {}
        default_transforms = preset.get("transforms") or {}

        self.reward_transforms = {**default_transforms, **self.reward_transforms}
        self.reward_groups = copy.deepcopy(default_groups)
        if reward_groups is not None:
            self.reward_groups = copy.deepcopy(reward_groups)
        self.routed_loss_weights = copy.deepcopy(default_routing)
        if routed_loss_weights is not None:
            self.routed_loss_weights = copy.deepcopy(routed_loss_weights)
        self.constraint_specs = copy.deepcopy(default_constraints)
        if constraint_specs is not None:
            self.constraint_specs = copy.deepcopy(constraint_specs)

        if group_coefficients is not None:
            for group_name, coeff in group_coefficients.items():
                self.reward_groups.setdefault(group_name, {}).setdefault("terms", {})
                self.reward_groups[group_name]["coefficient"] = coeff

        if not self.reward_routing_enabled:
            self.routed_loss_weights = self.routed_loss_weights or {}

        group_names = list(self.reward_groups.keys())
        if len(group_names) == 0:
            self.reward_groups = {"main": {"terms": {}, "coefficient": 1.0}}
            group_names = ["main"]

        self.group_names = group_names
        num_groups = len(group_names)
        self.group_name_to_idx = {name: idx for idx, name in enumerate(group_names)}

        self.register_buffer("time_bin_util_ema", torch.zeros(num_groups, self.time_num_bins))
        self.register_buffer("time_bin_logits", torch.zeros(num_groups, self.time_num_bins))
        self.register_buffer("time_bin_probs", torch.full((num_groups, self.time_num_bins), 1.0 / self.time_num_bins))

        constraint_names = list(self.constraint_specs.keys())
        self.constraint_names = constraint_names
        self.constraint_name_to_idx = {name: idx for idx, name in enumerate(constraint_names)}
        dual_init = [float(self.constraint_specs[name].get("weight_init", 0.0)) for name in constraint_names]
        dual_tensor = torch.tensor(dual_init, dtype=torch.float32) if dual_init else torch.zeros(0, dtype=torch.float32)
        self.register_buffer("constraint_lambdas", dual_tensor)

    def _init_optional_modules(self):
        self._optional_modules["rdkit_sascorer"] = self._load_optional_module("rdkit.Contrib.SA_Score.sascorer")
        posebusters_mod = self._load_optional_module("posebusters")
        if posebusters_mod is not None and hasattr(posebusters_mod, "PoseBusters"):
            try:
                self._posebusters = posebusters_mod.PoseBusters()
            except Exception:
                self._warn_once("posebusters_init", "PoseBusters found but failed to initialize; disabling PoseBusters metrics.")
                self._posebusters = None
        else:
            self._posebusters = None

    def _load_optional_module(self, module_name):
        if module_name in self._optional_modules:
            return self._optional_modules[module_name]
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module = None
        self._optional_modules[module_name] = module
        return module

    def _warn_once(self, key: str, message: str):
        if (not self.property_warning_once) or (key not in self._warned_messages):
            print(message)
            self._warned_messages.add(key)

    def _load_task_preset(self, preset_name):
        if preset_name is None:
            return {}
        return get_task_preset(preset_name)

    def _mol_to_fp(self, mol):
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    def _normalize_signal(self, signal: torch.Tensor):
        return (signal - signal.mean()) / (signal.std(unbiased=False) + self.reward_norm_eps)

    def _get_batch_reference_mols(self, batch, mols):
        if batch is None:
            return [self._global_reference_mol for _ in mols]
        refs = None
        for key in ["reference_mols", "reference_molecules", "ref_mols"]:
            if key in batch:
                refs = batch[key]
                break
        if refs is not None:
            refs = list(refs)
            if len(refs) < len(mols):
                refs = refs + [self._global_reference_mol] * (len(mols) - len(refs))
            return refs[: len(mols)]

        ref_smiles = None
        for key in ["reference_smiles", "ref_smiles"]:
            if key in batch:
                ref_smiles = batch[key]
                break
        if ref_smiles is not None:
            refs = []
            for smi in ref_smiles:
                if isinstance(smi, str):
                    refs.append(Chem.MolFromSmiles(smi))
                else:
                    refs.append(self._global_reference_mol)
            if len(refs) < len(mols):
                refs = refs + [self._global_reference_mol] * (len(mols) - len(refs))
            return refs[: len(mols)]
        return [self._global_reference_mol for _ in mols]

    def _safe_evaluator_tensor(self, name, mols, batch, dtype, device):
        evaluator = self.reward_evaluators.get(name)
        if evaluator is None:
            self._warn_once(name, f"Reward evaluator '{name}' not provided; returning zeros.")
            return torch.zeros(len(mols), dtype=dtype, device=device)
        values = evaluator(mols=mols, batch=batch)
        if isinstance(values, torch.Tensor):
            return values.to(device=device, dtype=dtype)
        return torch.tensor(values, dtype=dtype, device=device)

    def _compute_sa_score(self, mol):
        if mol is None:
            return None
        sascorer = self._optional_modules.get("rdkit_sascorer")
        if sascorer is None:
            return None
        return float(sascorer.calculateScore(mol))

    def _compute_strain_per_atom(self, mol):
        if mol is None:
            return None
        opt_mol = optimise_mol(mol)
        if opt_mol is None:
            return None
        opt_energy = calc_energy(opt_mol, per_atom=True)
        base_energy = calc_energy(mol, per_atom=True)
        if opt_energy is None or base_energy is None:
            return None
        return float(base_energy - opt_energy)

    def _smarts_indicator(self, mol, smarts_list, require_all=True, negate=False):
        if mol is None:
            return 0.0
        if len(smarts_list) == 0:
            return 1.0
        matches = []
        for pattern in smarts_list:
            query = Chem.MolFromSmarts(pattern)
            if query is None:
                continue
            matches.append(bool(mol.HasSubstructMatch(query)))
        if len(matches) == 0:
            return 1.0
        value = float(all(matches) if require_all else any(matches))
        return 1.0 - value if negate else value

    def _compute_reward_terms_from_mols(self, mols, batch, dtype, device):
        total = len(mols)
        refs = self._get_batch_reference_mols(batch, mols)
        raw_terms = {
            "validity_indicator": [],
            "connected_validity_indicator": [],
            "n_valid_atoms": [],
            "heavy_atom_count": [],
            "strain_energy_per_atom": [],
            "posebusters_validity_indicator": [],
            "qed": [],
            "tpsa": [],
            "logp": [],
            "molecular_weight": [],
            "tanimoto_similarity_to_reference": [],
            "sa_score_raw": [],
            "scaffold_similarity_to_reference": [],
            "murcko_scaffold_changed_indicator": [],
            "motif_retention_fraction": [],
            "required_smarts_satisfied_indicator": [],
            "forbidden_smarts_satisfied_indicator": [],
            "vina_score_raw": [],
            "shape_similarity_raw": [],
            "pharmacophore_match_raw": [],
            "clearance_pred_raw": [],
            "permeability_pred_raw": [],
            "solubility_pred_raw": [],
            "toxicity_pred_raw": [],
        }

        batch_motif_smarts = batch.get("motif_smarts") if batch is not None and isinstance(batch, dict) else None

        for idx, mol in enumerate(mols):
            ref_mol = refs[idx] if idx < len(refs) else self._global_reference_mol
            valid = float(mol is not None)
            connected = 0.0
            n_valid_atoms = 0.0
            heavy_atoms = 0.0
            strain = 0.0
            posebusters_valid = 0.0
            qed = 0.0
            tpsa = 0.0
            logp = 0.0
            mw = 0.0
            tanimoto = 0.0
            sa_score = 10.0
            scaffold_sim = 0.0
            scaffold_changed = 0.0
            motif_retention = 1.0
            required_smarts_ok = 1.0
            forbidden_smarts_ok = 1.0

            if mol is not None:
                try:
                    connected = float(len(Chem.GetMolFrags(mol)) == 1)
                except Exception:
                    connected = 0.0
                try:
                    n_valid_atoms = float(mol.GetNumAtoms())
                    heavy_atoms = float(Descriptors.HeavyAtomCount(mol))
                    qed = float(QED.qed(mol))
                    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                    logp = float(Crippen.MolLogP(mol))
                    mw = float(Descriptors.MolWt(mol))
                except Exception:
                    pass
                try:
                    strain_val = self._compute_strain_per_atom(mol)
                    if strain_val is not None:
                        strain = strain_val
                except Exception:
                    strain = 0.0
                try:
                    sa_val = self._compute_sa_score(mol)
                    if sa_val is not None:
                        sa_score = sa_val
                except Exception:
                    sa_score = 10.0
                if self._posebusters is not None:
                    try:
                        posebusters_valid = float(bool(self._posebusters.bust(mol)))
                    except Exception:
                        posebusters_valid = 0.0
                else:
                    posebusters_valid = 0.0
                mol_fp = self._mol_to_fp(mol)
                ref_fp = self._mol_to_fp(ref_mol) if ref_mol is not None else self._global_reference_fp
                if mol_fp is not None and ref_fp is not None:
                    tanimoto = float(DataStructs.TanimotoSimilarity(mol_fp, ref_fp))
                try:
                    mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                    ref_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ref_mol) if ref_mol is not None else None
                    if ref_scaffold is not None:
                        scaffold_sim = float(mol_scaffold == ref_scaffold)
                        scaffold_changed = float(mol_scaffold != ref_scaffold)
                except Exception:
                    scaffold_sim = 0.0
                    scaffold_changed = 0.0
                if batch_motif_smarts is not None and idx < len(batch_motif_smarts):
                    motif_patterns = batch_motif_smarts[idx]
                    if isinstance(motif_patterns, str):
                        motif_patterns = [motif_patterns]
                    matches = []
                    for pattern in motif_patterns or []:
                        query = Chem.MolFromSmarts(pattern)
                        if query is not None:
                            matches.append(float(mol.HasSubstructMatch(query)))
                    if len(matches) > 0:
                        motif_retention = float(sum(matches) / len(matches))
                required_smarts_ok = self._smarts_indicator(mol, self.required_smarts, require_all=True, negate=False)
                forbidden_smarts_ok = self._smarts_indicator(mol, self.forbidden_smarts, require_all=False, negate=True)

            raw_terms["validity_indicator"].append(valid)
            raw_terms["connected_validity_indicator"].append(connected)
            raw_terms["n_valid_atoms"].append(n_valid_atoms)
            raw_terms["heavy_atom_count"].append(heavy_atoms)
            raw_terms["strain_energy_per_atom"].append(strain)
            raw_terms["posebusters_validity_indicator"].append(posebusters_valid)
            raw_terms["qed"].append(qed)
            raw_terms["tpsa"].append(tpsa)
            raw_terms["logp"].append(logp)
            raw_terms["molecular_weight"].append(mw)
            raw_terms["tanimoto_similarity_to_reference"].append(tanimoto)
            raw_terms["sa_score_raw"].append(sa_score)
            raw_terms["scaffold_similarity_to_reference"].append(scaffold_sim)
            raw_terms["murcko_scaffold_changed_indicator"].append(scaffold_changed)
            raw_terms["motif_retention_fraction"].append(motif_retention)
            raw_terms["required_smarts_satisfied_indicator"].append(required_smarts_ok)
            raw_terms["forbidden_smarts_satisfied_indicator"].append(forbidden_smarts_ok)

        raw_terms["vina_score_raw"] = self._safe_evaluator_tensor("vina_score_raw", mols, batch, dtype, device)
        raw_terms["shape_similarity_raw"] = self._safe_evaluator_tensor("shape_similarity_raw", mols, batch, dtype, device)
        raw_terms["pharmacophore_match_raw"] = self._safe_evaluator_tensor("pharmacophore_match_raw", mols, batch, dtype, device)
        raw_terms["clearance_pred_raw"] = self._safe_evaluator_tensor("clearance_pred_raw", mols, batch, dtype, device)
        raw_terms["permeability_pred_raw"] = self._safe_evaluator_tensor("permeability_pred_raw", mols, batch, dtype, device)
        raw_terms["solubility_pred_raw"] = self._safe_evaluator_tensor("solubility_pred_raw", mols, batch, dtype, device)
        raw_terms["toxicity_pred_raw"] = self._safe_evaluator_tensor("toxicity_pred_raw", mols, batch, dtype, device)

        for key, values in list(raw_terms.items()):
            if isinstance(values, torch.Tensor):
                raw_terms[key] = values.to(device=device, dtype=dtype)
            else:
                if len(values) != total:
                    values = list(values) + [0.0] * (total - len(values))
                raw_terms[key] = torch.tensor(values, dtype=dtype, device=device)
        return raw_terms

    def _window_reward(self, value, lower=None, upper=None, center=None, width=None, mode="triangle"):
        if mode == "triangle":
            if center is None:
                if lower is None or upper is None:
                    return torch.ones_like(value)
                center = 0.5 * (lower + upper)
                width = max(0.5 * (upper - lower), self.reward_norm_eps)
            width = max(float(width), self.reward_norm_eps)
            return torch.clamp(1.0 - torch.abs(value - center) / width, min=0.0, max=1.0)

        if mode == "double_sigmoid":
            scale = max(float(width if width is not None else 1.0), self.reward_norm_eps)
            if lower is None:
                lower_gate = torch.ones_like(value)
            else:
                lower_gate = torch.sigmoid((value - lower) / scale)
            if upper is None:
                upper_gate = torch.ones_like(value)
            else:
                upper_gate = torch.sigmoid((upper - value) / scale)
            return lower_gate * upper_gate

        raise ValueError(f"Unsupported window reward mode '{mode}'")

    def _build_lower_is_better_reward(self, value, scale=1.0, offset=0.0):
        scale = max(float(scale), self.reward_norm_eps)
        return torch.exp(-(value - offset) / scale)

    def _compute_transformed_reward_terms(self, raw_terms, batch, dtype, device):
        transformed = {key: val for key, val in raw_terms.items()}
        transformed["qed_reward"] = raw_terms["qed"]
        transformed["tanimoto_reward"] = raw_terms["tanimoto_similarity_to_reference"]
        transformed["shape_similarity_reward"] = raw_terms["shape_similarity_raw"]
        transformed["pharmacophore_match_reward"] = raw_terms["pharmacophore_match_raw"]
        transformed["sa_reward"] = self._build_lower_is_better_reward(raw_terms["sa_score_raw"], scale=3.0)
        transformed["strain_reward"] = self._build_lower_is_better_reward(raw_terms["strain_energy_per_atom"], scale=0.2)
        transformed["vina_reward"] = self._build_lower_is_better_reward(raw_terms["vina_score_raw"], scale=5.0, offset=-12.0)
        transformed["clearance_reward"] = self._window_reward(raw_terms["clearance_pred_raw"], lower=0.0, upper=0.5, width=0.1, mode="double_sigmoid")
        transformed["motif_retention_reward"] = raw_terms["motif_retention_fraction"]
        transformed["scaffold_change_reward"] = raw_terms["murcko_scaffold_changed_indicator"]
        transformed["required_smarts_reward"] = raw_terms["required_smarts_satisfied_indicator"]
        transformed["forbidden_smarts_reward"] = raw_terms["forbidden_smarts_satisfied_indicator"]
        transformed["tpsa_window_reward"] = self._window_reward(raw_terms["tpsa"], lower=20.0, upper=120.0, mode="triangle")
        transformed["logp_window_reward"] = self._window_reward(raw_terms["logp"], lower=1.0, upper=4.0, mode="triangle")
        transformed["mw_window_reward"] = self._window_reward(raw_terms["molecular_weight"], lower=250.0, upper=550.0, mode="triangle")

        for reward_name, cfg in self.reward_transforms.items():
            source = cfg.get("source")
            if source is None or source not in transformed:
                continue
            transformed[reward_name] = self._window_reward(
                transformed[source],
                lower=cfg.get("lower"),
                upper=cfg.get("upper"),
                center=cfg.get("center"),
                width=cfg.get("width"),
                mode=cfg.get("mode", "triangle"),
            )

        return {key: value.to(device=device, dtype=dtype) for key, value in transformed.items()}

    def _aggregate_reward_group_scores(self, reward_terms):
        group_scores = {}
        for group_name, group_cfg in self.reward_groups.items():
            terms = group_cfg.get("terms", {})
            if len(terms) == 0:
                if len(reward_terms) == 0:
                    group_scores[group_name] = torch.tensor([])
                else:
                    template = next(iter(reward_terms.values()))
                    group_scores[group_name] = torch.zeros_like(template)
                continue
            accum = None
            weight_sum = 0.0
            for term_name, weight in terms.items():
                if term_name not in reward_terms:
                    self._warn_once(term_name, f"Reward term '{term_name}' not available; skipping in group '{group_name}'.")
                    continue
                term = reward_terms[term_name] * float(weight)
                accum = term if accum is None else accum + term
                weight_sum += abs(float(weight))
            if accum is None:
                template = next(iter(reward_terms.values()))
                accum = torch.zeros_like(template)
            if weight_sum > 0:
                accum = accum / weight_sum
            group_scores[group_name] = accum
        return group_scores

    def _compute_constraint_violations(self, reward_terms):
        violations = {}
        for name, spec in self.constraint_specs.items():
            source = spec["source"]
            if source not in reward_terms:
                template = next(iter(reward_terms.values()))
                violations[name] = torch.zeros_like(template)
                continue
            value = reward_terms[source]
            threshold = float(spec.get("threshold", 0.0))
            if spec.get("type", "min") == "min":
                violations[name] = F.relu(threshold - value)
            else:
                violations[name] = F.relu(value - threshold)
        return violations

    def _apply_constraint_penalty(self, group_scores, violations):
        effective = {}
        penalty = None
        if len(violations) > 0 and self.constraints_enabled:
            violation_stack = []
            for name in self.constraint_names:
                violation = violations[name]
                if self.constraint_centering:
                    violation = self._normalize_signal(violation)
                violation_stack.append(violation)
            if len(violation_stack) > 0:
                stacked = torch.stack(violation_stack, dim=0)
                lambdas = self.constraint_lambdas.to(stacked.device, stacked.dtype).view(-1, 1)
                penalty = (lambdas * stacked).sum(dim=0)
        for group_name, score in group_scores.items():
            effective[group_name] = score if penalty is None else score - penalty
        return effective

    def _sample_group_times(self, group_name, batchsize, device, dtype):
        if (not self.adaptive_time_sampling) or (group_name not in self.group_name_to_idx):
            t = self.time_dist.sample((batchsize,)).to(device=device, dtype=dtype)
            bins = torch.clamp((t * self.time_num_bins).long(), max=self.time_num_bins - 1)
            return t, bins
        group_idx = self.group_name_to_idx[group_name]
        probs = self.time_bin_probs[group_idx].to(device=device, dtype=dtype)
        bins = torch.multinomial(probs, num_samples=batchsize, replacement=True)
        edges = torch.linspace(0.0, 1.0, self.time_num_bins + 1, device=device, dtype=dtype)
        lower = edges[bins]
        upper = edges[bins + 1]
        t = lower + (upper - lower) * torch.rand(batchsize, device=device, dtype=dtype)
        return t, bins

    def _compute_importance_correction(self, group_name, bins, dtype, device):
        corr = torch.ones_like(bins, dtype=dtype, device=device)
        if (not self.adaptive_time_sampling) or (not self.time_importance_correction):
            return corr
        group_idx = self.group_name_to_idx[group_name]
        probs = self.time_bin_probs[group_idx].to(device=device, dtype=dtype)
        corr = (1.0 / self.time_num_bins) / probs[bins]
        corr = corr.clamp(min=self.time_importance_clip_min, max=self.time_importance_clip_max)
        return corr

    def _compute_routed_group_loss_per_sample(self, per_sample_losses, group_name):
        if not self.reward_routing_enabled:
            return sum(per_sample_losses.values())
        weights = self.routed_loss_weights.get(group_name)
        if weights is None:
            return sum(per_sample_losses.values())
        routed = None
        for loss_name, loss_val in per_sample_losses.items():
            component = float(weights.get(loss_name, 0.0)) * loss_val
            routed = component if routed is None else routed + component
        if routed is None:
            routed = sum(per_sample_losses.values())
        return routed

    def _update_time_sampler_from_batch(self, group_logs):
        if not self.adaptive_time_sampling:
            return
        for group_name, info in group_logs.items():
            if group_name not in self.group_name_to_idx:
                continue
            bins = info["bins"]
            if bins.numel() == 0:
                continue
            utility = self._normalize_signal(info["scores"]) * self._normalize_signal(-info["routed_loss"])
            group_idx = self.group_name_to_idx[group_name]
            for bin_idx in range(self.time_num_bins):
                mask = bins == bin_idx
                if not torch.any(mask):
                    continue
                batch_utility = utility[mask].mean().detach()
                old = self.time_bin_util_ema[group_idx, bin_idx]
                self.time_bin_util_ema[group_idx, bin_idx] = (1.0 - self.time_ema_beta) * old + self.time_ema_beta * batch_utility
            self.time_bin_logits[group_idx] = self.time_bin_util_ema[group_idx]
            probs = torch.softmax(self.time_bin_logits[group_idx] / self.time_tau, dim=0)
            probs = torch.clamp(probs, min=self.time_prob_floor)
            self.time_bin_probs[group_idx] = probs / probs.sum().clamp_min(self.reward_norm_eps)

    def _update_dual_variables(self, violations):
        if (not self.constraints_enabled) or len(self.constraint_names) == 0:
            return
        for idx, name in enumerate(self.constraint_names):
            violation_mean = violations[name].mean().detach().to(self.constraint_lambdas.device)
            updated = self.constraint_lambdas[idx] + self.dual_lr * violation_mean
            self.constraint_lambdas[idx] = torch.clamp(updated, min=0.0, max=self.dual_max)

    def _compute_rewards_from_generated(self, generated, batch=None):
        mols = self._generate_mols(generated, sanitise=True)
        dtype = generated["coords"].dtype
        device = generated["coords"].device
        raw_terms = self._compute_reward_terms_from_mols(mols, batch=batch, dtype=dtype, device=device)
        transformed_terms = self._compute_transformed_reward_terms(raw_terms, batch=batch, dtype=dtype, device=device)

        if self.task_preset_name is None and self.reward_groups == {"main": {"terms": {f"{self.reward_name}_reward": 1.0}, "coefficient": 1.0}}:
            reward_key = self.reward_name if self.reward_name in transformed_terms else f"{self.reward_name}_reward"
            rewards = transformed_terms.get(reward_key, transformed_terms.get("qed_reward", raw_terms["qed"]))
        else:
            rewards = next(iter(self._aggregate_reward_group_scores(transformed_terms).values()))
        return rewards, mols, raw_terms, transformed_terms

    def FM_training_step(self, batch):
        noise = self._build_noise_batch(batch)
        with torch.no_grad():
            generated = self._generate(
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )

        rewards, generated_mols, raw_terms, transformed_terms = self._compute_rewards_from_generated(generated, batch=batch)
        quality_metrics = self._compute_generation_quality_from_mols(
            generated_mols,
            dtype=generated["coords"].dtype,
            device=generated["coords"].device,
        )
        group_scores = self._aggregate_reward_group_scores(transformed_terms)
        violations = self._compute_constraint_violations({**raw_terms, **transformed_terms})
        effective_group_scores = self._apply_constraint_penalty(group_scores, violations)

        train_batch = self._build_generated_target_batch(batch, generated)
        batchsize = train_batch["natoms"].size(0)
        device = train_batch["real_coords"].device
        dtype = train_batch["real_coords"].dtype
        flag_3Ds = train_batch["flag_3Ds"]

        if self.formulation == "endpoint":
            coords_target = train_batch["real_coords"]
        else:
            coords_target = train_batch["real_coords"] - train_batch["noise_coords"]
        target = {
            "coords": coords_target,
            "atomics": train_batch["real_atomics"],
            "bonds": train_batch["real_bonds"],
            "charges": train_batch["real_charges"],
            "masks": train_batch["masks"],
        }

        group_losses = {}
        group_anchor_losses = []
        raw_loss_logs = {}
        group_logs = {}

        for group_name, group_cfg in self.reward_groups.items():
            coefficient = float(group_cfg.get("coefficient", 1.0))
            scores = effective_group_scores.get(group_name)
            if scores is None or scores.numel() == 0 or coefficient == 0.0:
                continue

            t, bins = self._sample_group_times(group_name, batchsize, device, dtype)
            interp_data = self.interpolate(train_batch, t, flag_3Ds=flag_3Ds)

            cond_batch = None
            if self.self_cond:
                cond_batch = {
                    "coords": torch.zeros_like(interp_data["coords"]),
                    "atomics": torch.zeros_like(interp_data["atomics"]),
                    "bonds": torch.zeros_like(interp_data["bonds"]),
                }
                if torch.rand(1).item() > 0.5:
                    with torch.no_grad():
                        cond_coords, cond_types, cond_bonds, _ = self(
                            interp_data,
                            t,
                            training=True,
                            cond_batch=cond_batch,
                            flag_3Ds=flag_3Ds,
                        )
                        cond_batch = {
                            "coords": cond_coords * flag_3Ds.view(-1, 1, 1),
                            "atomics": F.softmax(cond_types, dim=-1),
                            "bonds": F.softmax(cond_bonds, dim=-1),
                        }

            coords, types, bonds, charges = self(
                interp_data,
                t,
                training=True,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )
            predicted = {"coords": coords, "atomics": types, "bonds": bonds, "charges": charges}
            losses = self._loss_per_sample(target, predicted, flag_3Ds=flag_3Ds)
            routed_loss = self._compute_routed_group_loss_per_sample(losses, group_name)
            weights = self._reward_to_weights(scores)
            importance = self._compute_importance_correction(group_name, bins, dtype=dtype, device=device)
            weights = weights * importance
            weights = weights / weights.mean().clamp_min(self.reward_norm_eps)
            group_loss = (weights * routed_loss).mean()
            group_losses[group_name] = coefficient * group_loss
            group_logs[group_name] = {"bins": bins.detach(), "scores": scores.detach(), "routed_loss": routed_loss.detach()}

            for loss_name, loss_values in losses.items():
                raw_loss_logs.setdefault(loss_name, []).append(loss_values.mean())

            anchor_losses = self._anchor_loss_per_sample(
                interp_data,
                t,
                predicted,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )
            anchor_group = (
                anchor_losses["coord"]
                + anchor_losses["types"]
                + anchor_losses["bonds"]
                + anchor_losses["charges"]
            ).mean()
            group_anchor_losses.append(anchor_group)

            self.log(f"train-rl-{group_name}-score-mean", scores.mean(), on_step=True, logger=True, sync_dist=True)
            self.log(f"train-rl-{group_name}-weight-mean", weights.mean(), on_step=True, logger=True, sync_dist=True)
            self.log(f"train-rl-{group_name}-weight-max", weights.max(), on_step=True, logger=True, sync_dist=True)
            self.log(f"train-rl-{group_name}-routed-loss", routed_loss.mean(), on_step=True, logger=True, sync_dist=True)

        fm_loss = sum(group_losses.values()) if len(group_losses) > 0 else rewards.new_tensor(0.0)
        anchor_loss = torch.stack(group_anchor_losses).mean() if len(group_anchor_losses) > 0 else rewards.new_tensor(0.0)
        total_loss = fm_loss + (self.anchor_weight * self.anchor_loss_weight) * anchor_loss

        self._update_time_sampler_from_batch(group_logs)
        self._update_dual_variables(violations)

        for name, values in raw_loss_logs.items():
            self.log(f"train-fm-raw-{name}", torch.stack(values).mean(), on_step=True, logger=True, sync_dist=True)
        for group_name, group_loss in group_losses.items():
            self.log(f"train-fm-{group_name}", group_loss, on_step=True, logger=True, sync_dist=True)
        for name, violation in violations.items():
            self.log(f"train-constraint-{name}-violation-mean", violation.mean(), on_step=True, logger=True, sync_dist=True)
            if name in self.constraint_name_to_idx:
                lambda_val = self.constraint_lambdas[self.constraint_name_to_idx[name]]
                self.log(f"train-constraint-{name}-lambda", lambda_val, on_step=True, logger=True, sync_dist=True)
        for group_name in self.group_names:
            if group_name not in self.group_name_to_idx:
                continue
            group_idx = self.group_name_to_idx[group_name]
            for bin_idx in range(self.time_num_bins):
                self.log(f"train-timeprob-{group_name}-bin{bin_idx}", self.time_bin_probs[group_idx, bin_idx], on_step=True, logger=True, sync_dist=True)
                self.log(f"train-timeutil-{group_name}-bin{bin_idx}", self.time_bin_util_ema[group_idx, bin_idx], on_step=True, logger=True, sync_dist=True)
                if group_name in group_logs:
                    count = (group_logs[group_name]["bins"] == bin_idx).sum().float()
                    self.log(f"train-timecount-{group_name}-bin{bin_idx}", count, on_step=True, logger=True, sync_dist=True)

        self.log("train-rl-reward-mean", rewards.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-max", rewards.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-anchor-loss", anchor_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-fm-loss", fm_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-total-loss", total_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-n-valid", quality_metrics["n-valid"], on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-n-total", quality_metrics["n-total"], on_step=True, logger=True, sync_dist=True)

        return total_loss
