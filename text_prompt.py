import torch
import clip
import pandas as pd

def text_prompt():
    # Read labels from CSV file and convert to list
    classes_all = pd.read_csv('/home/mtech/project_env/Hyperformer/data/ntu60_labels.csv').values.tolist()

    # Define text prompt
    text_aug = 'This is a video about {}'

    # Tokenize the formatted labels
    classes_tokenized = torch.cat([clip.tokenize(text_aug.format(name)) for _, name in classes_all])

    return classes_tokenized
