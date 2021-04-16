from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

import pandas as pd

import torch
import torch.distributions
import torch.utils.data
from statsmodels.stats.proportion import binom_test

from dataset import SemanticsEvalDataset
from models.image_sentence_ranking.ranking_model import ImageSentenceRanker, cosine_sim
from preprocess import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
    DATA_PATH,
)
from utils import decode_caption, SEMANTICS_EVAL_FILES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_MAX_SAMPLES = 500


def get_semantics_eval_dataloader(eval_file, vocab):
    return torch.utils.data.DataLoader(
        SemanticsEvalDataset(
            DATA_PATH,
            IMAGES_FILENAME["test"],
            CAPTIONS_FILENAME["test"],
            eval_file,
            vocab,
        ),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )


def eval_semantics_score(model, dataloader, vocab, verbose=False):
    model.eval()

    accuracies = []
    with torch.no_grad():
        for batch_idx, (img, target_caption, distractor_caption) in enumerate(
            dataloader
        ):
            images = torch.cat((img, img))
            captions = torch.cat((target_caption, distractor_caption))
            caption_lengths = torch.tensor(
                [target_caption.shape[1], distractor_caption.shape[1]], device=device
            )

            if verbose:
                print(f"Target    : {decode_caption(target_caption[0], vocab)}")
                print(f"Distractor: {decode_caption(distractor_caption[0], vocab)}")


            images_embedded, captions_embedded = model(
                images, captions, caption_lengths
            )

            similarities = cosine_sim(images_embedded, captions_embedded)[0]

            if verbose:
                print(f"Similarity target    : {similarities[0]}")
                print(f"Similarity distractor: {similarities[1]}")

            if similarities[0] > similarities[1]:
                accuracies.append(1)
            elif similarities[0] < similarities[1]:
                accuracies.append(0)

            if len(accuracies) > EVAL_MAX_SAMPLES:
                break

    return accuracies


def main(args):
    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    print("Loading model checkpoint from {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)

    print("Loading image sentence ranking model.")
    word_embedding_size = 100
    joint_embeddings_size = 512
    lstm_hidden_size = 512
    model = ImageSentenceRanker(
        word_embedding_size,
        joint_embeddings_size,
        lstm_hidden_size,
        len(vocab),
        fine_tune_resnet=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    semantics_eval_loaders = {
        file: get_semantics_eval_dataloader(file, vocab)
        for file in SEMANTICS_EVAL_FILES
    }

    semantic_accuracies = {}
    for name, semantic_images_loader in semantics_eval_loaders.items():
        accuracies = eval_semantics_score(
            model, semantic_images_loader, vocab, verbose=args.verbose
        )
        mean_acc = np.mean(accuracies)
        p_value = binom_test(sum(accuracies), len(accuracies), alternative="larger")
        pd.DataFrame(accuracies).to_csv(name.replace("data", "results"), index=False)
        print(f"Accuracy for {name}: {mean_acc:.3f} p={p_value}\n")
        semantic_accuracies[name] = mean_acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("Start eval on device: ", device)
    args = get_args()
    main(args)
