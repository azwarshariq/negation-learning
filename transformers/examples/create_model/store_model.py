import argparse
import torch
from transformers.modeling_bert import BertConfig, BertForNegPreTraining

def load_and_save_model(config_path, model_weights_path, output_path):
    # Load the configuration
    config = BertConfig.from_pretrained(config_path)

    # Load the model with the fine-tuned weights
    model = BertForNegPreTraining(config)
    model.load_state_dict(torch.load(model_weights_path))

    # Save the model's state_dict
    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save BERT model")
    parser.add_argument("--config_path", required=True, help="Path to the configuration file")
    parser.add_argument("--model_weights_path", required=True, help="Path to the model weights file")
    parser.add_argument("--output_path", required=True, help="Path to save the output model")

    args = parser.parse_args()

    load_and_save_model(args.config_path, args.model_weights_path, args.output_path)
