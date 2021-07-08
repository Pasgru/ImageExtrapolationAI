import pickle
import torch

data_dir = r'C:\Users\Pascal\Pictures\dataset\DATA.pkl'


class ImgDataset(torch.utils.data.Dataset):
    """
    Latch data set comprising patterns as one-hot encoded instances.
    """

    def __init__(self):
        with open(data_dir, 'rb') as handle:
            data = pickle.load(handle)

        self.input_arrays = data['input_arrays']
        self.known_arrays = data['known_arrays']
        self.target_arrays = data['target_arrays']
        self.borders_x = data['borders_x']
        self.borders_y = data['borders_y']
        self.sample_ids = data['sample_ids']
        self.target_img_arrays = []

        for i in range(len(self.input_arrays)):
            target_img_array = self.input_arrays[i].copy()
            target_img_array[self.known_arrays[i] == 0] = self.target_arrays[i]
            self.target_img_arrays.append(target_img_array)

    def __len__(self) -> int:
        """
        Fetch amount of samples.

        :return: amount of samples
        """
        return len(self.input_arrays)

    def __getitem__(self, item_index: int):
        """
        Fetch specific sample.

        :param item_index: specific sample to fetch
        :return: specific sample as tuple of tensors
        """
        return (self.input_arrays[item_index].flatten(),
                self.target_img_arrays[item_index].flatten(),
                self.known_arrays[item_index].flatten())
