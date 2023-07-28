import torch
import pandas as pd
from collections import OrderedDict
import fire
from models import SmallModel, ConvClassifier2, ModelWithDropout, LargerModel

def get_model_summary(model, input_size):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_summary = OrderedDict()
    model_summary['Model Name'] = model.__class__.__name__
    model_summary['Total Parameters'] = total_params
    model_summary['Trainable Parameters'] = trainable_params
    model_summary['Non-trainable Parameters'] = total_params - trainable_params

    return model_summary

def generate_model_summary(alignment_length, num_classes, output):
    models = [SmallModel(alignment_length,num_classes), ConvClassifier2(alignment_length,num_classes), ModelWithDropout(alignment_length,num_classes), LargerModel(alignment_length,num_classes)]
    model_summaries = []

    for model in models:
        model_summaries.append(get_model_summary(model, (6,alignment_length)))

    df = pd.DataFrame(model_summaries)
    df.to_csv(output, index=False)
    print(df)

if __name__ == "__main__":
    fire.Fire(generate_model_summary)
