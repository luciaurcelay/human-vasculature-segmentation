import os

from dataset import create_dataset
from data_loader import create_dataloaders
from utils.argument_parser import create_parser

def main():
     args = create_parser()

     # Create custom dataset
     if args.create_dataset:
          create_dataset()

     # Train model
     if args.train:
          print("hi, you are calling train function")
          train_loader, val_loader, test_loader = create_dataloaders()

     # Test model
     if args.test:
          print("hi, you are calling test function")
     

if __name__ == "__main__":
    main()
