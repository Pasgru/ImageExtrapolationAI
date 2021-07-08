import torch


class CNN(torch.nn.Module):
    """
    Yes
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.lin1 = torch.nn.Linear(90*90, 90*90)
        self.ac1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(90*90, 90*90)
        self.ac2 = torch.nn.ReLU()
        self.lin3 = torch.nn.Linear(90*90, 90*90)
        self.ac3 = torch.nn.ReLU()
        self.lin4 = torch.nn.Linear(90*90, 90*90)
        self.ac4 = torch.nn.ReLU()
        self.lin5 = torch.nn.Linear(90*90, 90*90)
        self.ac5 = torch.nn.ReLU()
        self.lin6 = torch.nn.Linear(90*90, 90*90)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.lin1(x)
        x = self.ac1(x)
        x = self.lin2(x)
        x = self.ac2(x)
        x = self.lin3(x)
        x = self.ac3(x)
        x = self.lin4(x)
        x = self.ac4(x)
        x = self.lin5(x)
        x = self.ac5(x)

        return self.lin6(x)
