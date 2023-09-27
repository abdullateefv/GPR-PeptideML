import csv
import logging
import random
import warnings

import numpy as np
import pandas as pd
from pyrosetta import *
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import OperateOnResidueSubset, PreventRepackingRLT, \
    ExtraRotamersGenericRLT
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover, MinMover
from sklearn.gaussian_process import GaussianProcessRegressor

import modelFiles

logger = logging.getLogger(__name__)

MAX_ITER = 2
SAMPLE_SIZE = 10000


def initialTraining(model):
    data = pd.read_csv("../data/VEGFRFullData_TEST.csv")
    X = data["Peptide Sequences"].apply(modelFiles.extract_features)
    Y = data["Binding Affinities"]
    X = pd.DataFrame(X.to_list()).values

    model.fit(X, Y)
    return model


def reinforce(model):
    print("\033[1mPeptideML: \033[0m" + "\033[93m" + "Reinforcing model with new data..." + "\033[0m", end="")
    initialTraining(model)
    print("\r" + "\033[1mPeptideML: \033[0m" + "\033[92m" + "Reinforced model with new data" + "\033[0m")
    return model


def getMostUncertain(model, sampleSize):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
    greatestUncertainty = 0
    mostUncertainPeptide = ''
    for samples in range(sampleSize):
        samplePeptide = ''.join(random.choice(amino_acids) for _ in range(16))
        y_pred, sampleUncertainty = model.predict(np.array([[samplePeptide]]), return_std=True)
        if (sampleUncertainty[0] > greatestUncertainty):
            greatestUncertainty = sampleUncertainty[0]
            mostUncertainPeptide = samplePeptide

        progress = (samples + 1) / sampleSize
        bar_length = 50
        progress_chars = int(progress * bar_length)
        progress_bar = '\033[93m' + '[' + '█' * progress_chars + '\033[0m░' * (bar_length - progress_chars) + ']'
        print('\r' + "\033[1mPeptideML: \033[0m" + str(
            samples + 1) + " sequences evaluated for model certainty | " + progress_bar + f" {progress * 100:.2f}% complete",
              end="")
    z = 2 + 1
    print('\033[0m')
    print("\033[1mPeptideML: \033[0m" + "Most uncertain peptide was: " + mostUncertainPeptide)
    return mostUncertainPeptide


def pyRosetta(mostUncertain):
    # Load the pose and make a copy to work on
    original_pose = pose_from_pdb("../data/pose.pdb")
    pose = Pose()
    pose.assign(original_pose)

    # Ensure the pose has two chains
    if pose.num_chains() < 2:
        raise ValueError("The input pose must have at least two chains.")

    # Generate a Pose for the new peptide sequence
    peptide_pose = pose_from_sequence(mostUncertain)

    # Delete the original chain B from the main Pose
    chain_B_start = pose.chain_begin(2)
    chain_B_end = pose.chain_end(2)
    pose.delete_residue_range_slow(chain_B_start, chain_B_end)

    # Append the new peptide to the original Pose
    for i in range(1, peptide_pose.total_residue() + 1):
        pose.append_residue_by_jump(peptide_pose.residue(i), pose.total_residue())

    # Set up residue selectors
    protein_selector = ChainSelector("A")
    peptide_selector = ChainSelector("B")

    # Set up task operations
    repack_protein = OperateOnResidueSubset(PreventRepackingRLT(), protein_selector)
    extrarot_peptide = OperateOnResidueSubset(ExtraRotamersGenericRLT(), peptide_selector)

    tf = TaskFactory()
    tf.push_back(repack_protein)
    tf.push_back(extrarot_peptide)

    packer_task = tf.create_task_and_apply_taskoperations(pose)

    # Set up the ScoreFunction
    scorefxn = create_score_function("ref2015")

    # Pack the pose with the new peptide sequence
    pack_mover = PackRotamersMover(scorefxn, packer_task)
    pack_mover.apply(pose)

    # Minimize the pose to further improve the score
    min_mover = MinMover()
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn)
    min_mover.apply(pose)

    # Return the computed score
    score = scorefxn(pose)

    if score <= 0:
        print(
            "\033[1mPeptideML: \033[0m" + "Pyrosetta reported score for most uncertain peptide of: " + "\033[32m" + str(
                score) + "\033[0m")
    else:
        print(
            "\033[1mPeptideML: \033[0m" + "PyRosetta reported score for most uncertain peptide of: " + "\033[31m" + str(
                score) + "\033[0m")
    return score


def updateTrainingData(mostUncertain, score):
    with open("../data/VEGFRFullData_TEST.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([mostUncertain, score])
    print("\033[1mPeptideML: \033[0m" + "Updated the training data with sequence: " + str(
        mostUncertain) + " and score: " + str(score))


def initializeReinforcement(model, maxIter):
    print("\033[1mPeptideML: \033[0m" + "\033[93m" + "Training initial model..." + "\033[0m", end="")
    model = initialTraining(model)
    print("\r" + "\033[1mPeptideML: \033[0m" + "\033[92mTrained initial model" + "\033[0m")

    for iter in range(maxIter):
        print("__________\n" + "\033[1mPeptideML: \033[0m" + "Beginning reinforcement iteration: " + str(
            iter + 1) + "\033[0m")
        mostUncertain = getMostUncertain(model, sampleSize=SAMPLE_SIZE)
        score = pyRosetta(mostUncertain)
        updateTrainingData(mostUncertain, score)
        reinforce(model)


def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    logger.warning('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))


def setupWarningsLogger():
    # Step 1: Set up the logging module
    logger.setLevel(logging.WARNING)

    # Create a file handler and set its level to WARNING
    file_handler = logging.FileHandler('warnings.log')
    file_handler.setLevel(logging.WARNING)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Step 2: Redirect warnings to the logging system
    warnings.showwarning = custom_showwarning

    # Example warning
    warnings.warn("This is a test warning!")


pyrosetta.init(options="-mute all -no_output")
setupWarningsLogger()
print("\033[1mPeptideML: \033[0m" + "Beginning reinforcement with \033[1m" + str(
    MAX_ITER) + "\033[0m maximum reinforcement interations and uncertainty sampling size of \033[1m" + str(
    SAMPLE_SIZE) + "\033[0m")
model = GaussianProcessRegressor(kernel=modelFiles.SequenceKernel(), alpha=0.1)
initializeReinforcement(model, maxIter=MAX_ITER)
print("\033[1mPeptideML: \033[0m" + "\033[92mCompleted")
