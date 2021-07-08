"""
Author: <Pascal Gruber>
Matr.Nr.: <12005914>
Exercise <5>
"""
import pickle
from PIL import Image
import numpy as np
import random
from model import CNN
from createDataset import ImgDataset
import torch


def random_img(input_arrays, known_arrays, border_x, border_y, sample_ids):
    target = input_arrays[known_arrays == 0]
    for i in range(len(target)):
        target[i] = random.randint(0, 255)
    return target


def mean_img(input_arrays, known_arrays, border_x, border_y, sample_ids):
    mean = np.mean(input_arrays)
    target = input_arrays[known_arrays == 0]
    target[:] = mean
    return target


def make_img(data):
    for i in data:
        test = i[0]
        img = Image.fromarray(i[0], 'L')
        img.save(f"images/{i[4]}.png")


def pkl_to_data(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)

    input_arrays = data['input_arrays']
    known_arrays = data['known_arrays']
    border_x = data['borders_x']
    border_y = data['borders_y']
    sample_ids = data['sample_ids']

    for i in range(len(input_arrays)):
        yield input_arrays[i], known_arrays[i], border_x[i], border_y[i], sample_ids[i]


def train_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = r'cpu') -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param model: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    :return: None
    """
    model.train()
    # Found here: https://www.programmersought.com/article/53493453409/
    criterion = torch.nn.MSELoss()
    for batch_index, (data, target, known) in enumerate(data_loader):
        data, target = data.float().to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device = r'cpu'):
    """
    Test specified network on specified data loader.

    :param model: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: cross-entropy loss as well as accuracy
    """
    model.eval()
    loss = 0.0
    correct = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for data, target, known in data_loader:
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)

            data = data.float().to(r'cpu')
            final_img = data.numpy()[0]
            print(final_img)
            final_img[known.numpy()[0] == 0] = output.float().to(r'cpu').numpy()[0][known.numpy()[0] == 0]
            final_img = torch.tensor(final_img).to(device)

            loss += float(criterion(final_img, target).item())
            # print(output.max())
            # pred = output.max(1, keepdim=True)[1]#
            # correct += int(pred.eq(target.view_as(pred)).sum().item())

    return loss / len(data_loader.dataset)  # , correct / len(data_loader.dataset)


if __name__ == "__main__":
    '''
    Random: -8356
    Mean: -3787
    
    file = "example_testset.pkl"
    data = pkl_to_data(file)

    targets = []
    for input_arrays, known_arrays, border_x, border_y, sample_ids in data:
        target = random_img(input_arrays, known_arrays, border_x, border_y, sample_ids)

        input_arrays[known_arrays == 0] = target
        img = Image.fromarray(input_arrays, 'L')
        img.save(f"outputIMG/{sample_ids}.png")

        targets.append(target)

    with open("solution.pkl", "wb") as submission:
        pickle.dump(targets, submission)
    '''

    model = CNN()
    cDataSet = ImgDataset()
    cDataSetTrain = torch.utils.data.Subset(cDataSet, list(range(100)))
    cDataSetTest = torch.utils.data.Subset(cDataSet, list(range(25000, len(cDataSet))))

    device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_set = torch.utils.data.DataLoader(
        dataset=cDataSetTrain,
        batch_size=1
    )
    test_set = torch.utils.data.DataLoader(
        dataset=cDataSetTest,
        batch_size=1
    )

    for epoch in range(1):
        train_network(model=model, data_loader=train_set, device=device, optimizer=optimizer)
        performance = test_network(model=model, data_loader=train_set, device=device)

    for input, target in test_set:
        input = input.float().to(device)
        output = model(input)
        print(output)
        output = output.int().to(r'cpu')
        print(output)
        Image.fromarray(np.uint8(output.reshape(90, 90)), 'L').show()
        break
