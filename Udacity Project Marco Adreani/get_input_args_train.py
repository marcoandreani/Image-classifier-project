import argparse

def get_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', action="store")

    parser.add_argument('--save_dir', action="store", default=".", dest='save_dir', type=str, help='Save directory')
    
    parser.add_argument('--categories_json', action="store", default="cat_to_name.json", dest='categories_json', type=str, help='Path to category file.')

    parser.add_argument('--arch', action="store", default="vgg16", dest='arch', type=str, help='Supported architectures: vgg16 , vgg13, densenet121')

    parser.add_argument('--gpu', action="store_true", dest="use_gpu", default=False, help='Use GPU')
    
    parser.add_argument('--learning_rate', action="store", default=0.01, type=float, help='Learning rate')

    parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=[512], type=int, help='Hidden layer units')

    parser.add_argument('--epochs', action="store", dest="epochs", default=20, type=int, help='Epochs')
    
    parser.add_argument('image_path', help='Path to image file.', action="store")
    
    parser.add_argument('checkpoint', help='Path to checkpoint file.', action="store")

    parser.add_argument('--save_dir', action="store", default=".", dest='save_dir', type=str, help='Directory to save training checkpoint file' )

    parser.add_argument('--top_k', action="store", default=5, dest='top_k', type=int, help='Return top K most likely classes.' )

    parser.parse_args()
    return parser


    


