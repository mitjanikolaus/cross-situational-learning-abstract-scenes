import os
import pickle

import nltk
import pandas as pd
import h5py as h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


from preprocess import (
    MEAN_ABSTRACT_SCENES,
    STD_ABSTRACT_SCENES,
    encode_caption,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    CAPTIONS_PER_IMAGE = 6

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        vocab,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(os.path.join(data_folder, features_filename), "r")

        self.features_scale_factor = features_scale_factor

        # Load captions
        with open(os.path.join(data_folder, captions_filename), "rb") as file:
            self.captions = pickle.load(file)

        # Set pytorch transformation pipeline
        self.normalize = transforms.Normalize(
            mean=MEAN_ABSTRACT_SCENES, std=STD_ABSTRACT_SCENES
        )

        self.image_ids = [int(i) for i in list(self.images.keys())]

        self.vocab = vocab

    def get_image_features(self, id, channels_first=True, normalize=True):
        image_data = self.images[str(id)][()]

        image = torch.FloatTensor(image_data)

        if channels_first:
            image = image.permute(2, 0, 1)

        if normalize:
            image = self.normalize(image)

        # scale the features with given factor
        image = image * self.features_scale_factor

        return image

    def __getitem__(self, i):
        image_id = self.image_ids[i // self.CAPTIONS_PER_IMAGE]
        caption_id = i % self.CAPTIONS_PER_IMAGE

        image = self.get_image_features(image_id)

        caption = self.captions[image_id][caption_id]

        caption = torch.LongTensor(caption)

        return image, caption, image_id

    def __len__(self):
        return len(self.images) * self.CAPTIONS_PER_IMAGE

    def pad_collate(batch):
        images = torch.stack([s[0] for s in batch])
        captions = [s[1] for s in batch]
        image_ids = torch.tensor([s[2] for s in batch])

        sequence_lengths = torch.tensor([len(c) for c in captions])
        padded_captions = pad_sequence(captions, batch_first=True)

        return (
            images.to(device),
            padded_captions.to(device),
            sequence_lengths.to(device),
            image_ids,
        )


class SemanticsEvalDataset(Dataset):
    """
    PyTorch Dataset that provides sets of target and distractor images for syntax learning evaluation
    """

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        eval_csv,
        vocab,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param data_indices: dataset split, indices of images that should be included
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(os.path.join(data_folder, features_filename), "r")

        self.features_scale_factor = features_scale_factor

        # Set pytorch transformation pipeline
        self.normalize = transforms.Normalize(
            mean=MEAN_ABSTRACT_SCENES, std=STD_ABSTRACT_SCENES
        )

        self.vocab = vocab

        self.data = pd.read_csv(eval_csv)

    def get_image_features(self, id, channels_first=True, normalize=True):
        image_data = self.images[str(id)][()]

        # show_image(image_data)

        image = torch.tensor(image_data, device=device, dtype=torch.float)

        if channels_first:
            image = image.permute(2, 0, 1)

        if normalize:
            image = self.normalize(image)

        # scale the features with given factor (convert values from [0, 256] to [0, 1]
        image = image * self.features_scale_factor

        return image

    def __getitem__(self, i):
        img_id, target_sentence, distractor_sentence = self.data.iloc[i]
        img = self.get_image_features(img_id)

        target_sentence = nltk.word_tokenize(target_sentence)
        target_sentence = encode_caption(target_sentence, self.vocab)
        target_sentence = torch.tensor(target_sentence, device=device)

        distractor_sentence = nltk.word_tokenize(distractor_sentence)
        distractor_sentence = encode_caption(distractor_sentence, self.vocab)
        distractor_sentence = torch.tensor(distractor_sentence, device=device)

        return img, target_sentence, distractor_sentence

    def __len__(self):
        length = len(self.data)

        return length
