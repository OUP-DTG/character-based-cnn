import argparse
import torch
import torch.nn.functional as F
from src.sentence_model import SentenceCNN
from src import utils
from train import SWISS_GERMAN_ALPHABET, NUM_SG_CHARS

use_cuda = torch.cuda.is_available()


def predict(args):
    model = SentenceCNN(args, args.number_of_classes)
    state = torch.load(args.model)
    model.load_state_dict(state)
    model.eval()

    processed_input = utils.preprocess_input(args)
    processed_input = torch.tensor(processed_input)
    processed_input = processed_input.unsqueeze(0)
    if use_cuda:
        processed_input = processed_input.to("cuda")
        model = model.to("cuda")
    prediction = model(processed_input)
    probabilities = F.softmax(prediction, dim=1)
    probabilities = probabilities.detach().cpu().numpy()
    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str, help="path for pre-trained model")
    parser.add_argument("--text", type=str, default="I love pizza!", help="text string")
    parser.add_argument("--steps", nargs="+", default=["lower"])

    # arguments needed for the predicition
    parser.add_argument(
        "--alphabet",
        type=str,
        default=SWISS_GERMAN_ALPHABET,
    )
    parser.add_argument("--number_of_characters", type=int, default=NUM_SG_CHARS)
    #parser.add_argument("--extra_characters", type=str, default="éàèùâêîôûçëïü")
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--number_of_classes", type=int, default=4)

    args = parser.parse_args()
    prediction = predict(args)

    print("input : {}".format(args.text))
    print("prediction : {}".format(prediction))
