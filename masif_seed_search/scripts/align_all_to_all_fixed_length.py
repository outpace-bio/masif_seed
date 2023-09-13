from Bio.PDB import StructureBuilder, PDBIO, PDBParser, Selection
from scipy.spatial import cKDTree
import numpy as np
import argparse
import os
import sys
import glob
import multiprocessing as mp
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


###### METHODS #########
# Save a fragment
def save_fragment(out_fn, residues, target_residues=None):
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()
    seed_chain = "A"
    if target_residues is not None:
        seed_chain = "B"
        structBuild.init_chain("A")
        for res in target_residues:
            outputStruct[0]["A"].add(res)
    structBuild.init_chain(seed_chain)
    for res in residues:
        outputStruct[0][seed_chain].add(res)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(out_fn)


# Return the fragment of residues touching the target dots.
def find_seed_helix(target_dots, residues, frag_size=12):
    ckd = cKDTree(target_dots)
    if len(residues) < frag_size:
        return None
    res_values = []
    for ix, res in enumerate(residues):
        vals_for_res = 0
        for atom in res:
            dist, r = ckd.query(atom.get_coord())
            if dist < 3.0:
                vals_for_res += 1.0
        res_values.append(np.sum(vals_for_res))
    res_values = np.array(res_values)
    best_frag = -1
    best_frag_val = -1
    for ix in range(len(residues) - frag_size + 1):
        if np.sum(res_values[ix : ix + frag_size]) > best_frag_val:
            best_frag_val = np.sum(res_values[ix : ix + frag_size])
            best_frag = ix
    return residues[best_frag : best_frag + frag_size]


def compute_helix_to_peptide_rmsd(ref_peptide, helix):
    # Compute nearest neighbors between the two peptides and the calpha to calpha rmsd.

    ref_coord = np.array([x["CA"].get_coord() for x in ref_peptide])
    helix_coord = np.array([x["CA"].get_coord() for x in helix])

    dists = np.sqrt(np.sum(np.square(ref_coord - helix_coord), axis=1))
    rmsd = np.sqrt(np.mean(np.square(dists)))
    return rmsd


###### END METHODS #########


def align_peptide_to_all_others(target, seed_path, all_seeds, pre_refine=False):
    all_dots = []
    with open(f"db_search_results/{target}/site_0/target.vert") as f:
        dots_lines = f.readlines()
        for point_line in dots_lines:
            point = [float(x) for x in point_line.split(",")]
            all_dots.append(point)

    all_dots = np.array(all_dots)
    current_struct_fn = seed_path
    current_pep = os.path.basename(current_struct_fn).replace("_refine.pdb", "")

    # Open original peptide.
    parser = PDBParser()
    orig_struct = parser.get_structure(current_struct_fn, current_struct_fn)[0]
    if not pre_refine:
        target_struct = orig_struct["A"]
        orig_struct = orig_struct["B"]
    orig_residues = Selection.unfold_entities(orig_struct, "R")

    # Find the amino acids touching target dots
    orig_fragment = find_seed_helix(all_dots, orig_residues)
    if orig_fragment is None:
        return

    save_fragment(
        f"analysis_fixed_size/{target}/out_pdb/{current_pep}_frag.pdb",
        orig_fragment,
        None if pre_refine else target_struct,
    )

    all_rmsd = []
    with open("alignment_errorfile.txt", "a") as f:
        for num_compared, mypep in enumerate(all_seeds, start=1):
            compared_struct = parser.get_structure(mypep, mypep)[0][
                "A" if pre_refine else "B"
            ]
            residues_compared_struct = Selection.unfold_entities(compared_struct, "R")

            # Find the amino acids touching target dots
            compared_fragment = find_seed_helix(all_dots, residues_compared_struct)
            if compared_fragment is None:
                all_rmsd.append(-1)
                continue

            # Do rmsds in both directions and take the minimum.
            rmsd1 = compute_helix_to_peptide_rmsd(orig_fragment, compared_fragment)
            rmsd2 = compute_helix_to_peptide_rmsd(compared_fragment, orig_fragment)

            rmsd = min([rmsd1, rmsd2])
            if rmsd > 100:
                f.write(f"ERROR: {rmsd} {mypep} {seed_path}\n")

            all_rmsd.append(rmsd)

    np.save(f"analysis_fixed_size/{target}/out_data/rmsd_{current_pep}", all_rmsd)


def main(args):
    os.makedirs(f"analysis_fixed_size/{args.target}/out_pdb/", exist_ok=True)
    os.makedirs(f"analysis_fixed_size/{args.target}/out_data", exist_ok=True)

    if args.pre_refine:
        all_seeds = glob.glob(f"db_search_results/{args.target}/*/*.pdb")
    else:
        all_seeds = glob.glob(
            f"refined_seeds/{args.target}/selected_seeds/*_refine.pdb"
        )
    all_seeds = sorted(all_seeds)
    with mp.Pool() as p:
        tasks = [
            p.apply_async(
                align_peptide_to_all_others,
                args=(args.target, seed, all_seeds[i + 1 :], args.pre_refine),
            )
            for i, seed in enumerate(all_seeds)
        ]
        for t in tqdm(tasks, desc="Aligning..."):
            t.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align all peptides to all others.")
    parser.add_argument("--target", type=str, help="Target name.")
    parser.add_argument(
        "--pre-refine",
        action="store_true",
        help="Tells the script to use the pre-refined peptides.",
    )
    args = parser.parse_args()

    main(args)
