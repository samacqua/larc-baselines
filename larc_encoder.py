import torch
import torch.nn as nn
from transformers import BertModel


class LARCEncoder(nn.Module):
    """
    context encoder of LARC task
    assumes 3 IO examples, where each is 30x30. assumes test input is 30x30. assumes description is tokenized by LM.
    https://files.slack.com/files-pri/T01044K0LBZ-F02D2MGQF62/screen_shot_2021-09-02_at_1.56.03_pm.png
    TODO:
        - better way to handle different grid sizes than padding with special token
        - make LM changeable at initialization
    """

    def __init__(self, max_grid_size=(30, 30), num_ios=3, use_nl=True):
        super().__init__()

        # h_out = (h_in + 2*padding[0] - dilation[0]*(kernel_size[0]−1) - 1) / stride[0] + 1
        # w_out = (w_in + 2*padding[1] - dilation[1]*(kernel_size[1]−1) - 1) / stride[1] + 1

        self.use_nl = use_nl

        # grid encoder
        # BxWxHx11 --> Bx256
        linear_input_size = 64 * max_grid_size[0] * max_grid_size[1]
        self.encoder = nn.Sequential(
            nn.Conv2d(11, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Flatten(),
            nn.Linear(linear_input_size, 256)
        )

        # input vs. output embedding
        # Bx256 --> Bx128
        self.in_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.out_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # example embedding
        # Bx256 --> Bx64
        self.ex_encoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # test input embedding
        # Bx256 --> Bx64
        self.test_in_embedding = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # natural language description encoding
        # BxNL --> Bx64
        if use_nl:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.bert.requires_grad_(False)
            self.bert_resize = nn.Sequential(
                nn.Linear(768, 64),
                nn.ReLU(),
            )

        # transformer
        # D = num_ios + (1 if use_nl else 0) + 1    (1 is for test input)
        # BxDx64 --> BxDx64
        transform_dimension = num_ios + int(use_nl) + 1
        encoder_layer = nn.TransformerEncoderLayer(d_model=transform_dimension, nhead=transform_dimension, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, io_grids, test_in, desc_tokens=None):
        # run grids through encoders
        transformer_input = []
        for io_in, io_out in io_grids:
            io_in = self.in_encoder(self.encoder(io_in))
            io_out = self.out_encoder(self.encoder(io_out))
            io = self.ex_encoder(torch.cat((io_in, io_out), dim=-1))

            transformer_input.append(io)

        # run test input grid through encoder
        transformer_input.append(self.test_in_embedding(self.encoder(test_in)))

        # run through BERT
        if self.use_nl:
            transformer_input.append(self.bert_resize(self.bert(**desc_tokens)['pooler_output']))

        # concatenate all inputs and run through transformer
        t_in = torch.stack(transformer_input).permute(1, 2, 0)
        t_out = self.transformer(t_in)

        return t_out

if __name__ == '__main__':
    from transformers import BertTokenizer
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    max_w, max_h = 10, 10
    ios = [(torch.zeros((batch_size, 11, max_h, max_w), device=DEVICE), torch.zeros((batch_size, 11, max_h, max_w), device=DEVICE))
           for _ in range(3)]
    test_in = torch.zeros((batch_size, 11, max_h, max_w), device=DEVICE)
    descriptions = ["flip square and make it blue."] * (batch_size - 1) + [
        "turn it yellow."]  # to make sure can accept variable description lengths

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    desc_enc = {k: torch.tensor(v, device=DEVICE) for k, v in
                tokenizer.batch_encode_plus(descriptions, padding=True).items()}

    predictor = LARCEncoder(max_grid_size=(max_w, max_h)).to(DEVICE)
    res = predictor(ios, test_in, desc_enc)
    print(res.shape)