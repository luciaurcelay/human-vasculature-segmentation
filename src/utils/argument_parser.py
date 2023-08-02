import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Human vasculature segmentation")

    # Add arguments for different functionalities
    parser.add_argument("-d", "--create_dataset", action="store_true", help="Create the dataset")
    parser.add_argument("-tr", "--train", action="store_true", help="Train the model")
    parser.add_argument("-te", "--test", action="store_true", help="Test the model")
    parser.add_argument("-s", "--save model", action="store_false")

    # Add arguments for hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    return parser.parse_args()