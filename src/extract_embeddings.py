import argparse
import torch
from src.sentence_model import SentenceCNN
import os
from src import utils
from torch.utils.data import DataLoader


def extract_embeddings_from_trained_model(args):
    def _up_n(path, n):
        components = os.path.normpath(path).split(os.sep)
        return os.sep.join(components[:-n])

    model = SentenceCNN(args, args.number_of_classes)
    state = torch.load(os.path.join(_up_n(os.path.dirname(__file__), 1), 'models', args.model))
    model.load_state_dict(state)

    embeddings = []

    data_generator = DataLoader(validation_set, **validation_params)
    sentence = "mir sind alli di chliine gschwüschterti sind zu verwante choo"
    sentence = utils.process_text(["lower"], sentence)

    model.eval()
    model.share_memory()  # NOTE: this is required for the ``fork`` method to work
    with torch.no_grad():

        outputs = model(sentence)


    layer = model._modules.get('conv_layers')[-2]  # access the penultimate layer
    # layer = model._modules
    # for child in model.named_children():
    #     print(child)
    # print(model.named_children())
    # print(type(layer.get('conv_layers')[-2]))
    # print(type(layer.get('conv_layers')[-2]))
    _ = layer.register_forward_hook(copy_embeddings)
    print(embeddings)
    return embeddings


def copy_embeddings(m, i, o):
    """Copy embeddings from the penultimate layer."""
    o = o[:, :, 0, 0].detach().numpy().tolist()
    embeddings.append(o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str, help="path for pre-trained model",
                        default="model_test_model_epoch_9_maxlen_150_lr_0.01_loss_0.7056_acc_0.7302_f1_0.7184.pth")

    # arguments needed for the model
    parser.add_argument(
        "--alphabet",
        type=str,
        default="() *,-./0123456789?ABCDEFGHIJKLMNOPRSTUVWZ_abcdefghijklmnoprstuvwxyzàáãäèéìíòóõöùúüĩǜ̀́ẽ",
    )
    parser.add_argument("--number_of_characters", type=int, default=88)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--number_of_classes", type=int, default=4)

    args = parser.parse_args()
    embeddings = extract_embeddings_from_trained_model(args)
    print(embeddings)
