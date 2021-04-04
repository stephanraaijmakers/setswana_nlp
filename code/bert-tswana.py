#! pip install tokenizers

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = "./tswana_data/train.txt"


# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=50000, min_frequency=2, special_tokens=[
    "<s>",		     
    "<pad>",	     
    "</s>",	     
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("tswana_models")

import json
config = {
	"architectures": [
		"RobertaForMaskedLM"
	],
	"attention_probs_dropout_prob": 0.1,
	"hidden_act": "gelu",
	"hidden_dropout_prob": 0.1,
	"hidden_size": 768,
	"initializer_range": 0.02,
	"intermediate_size": 3072,
	"layer_norm_eps": 1e-05,
	"max_position_embeddings": 514,
	"model_type": "roberta",
	"num_attention_heads": 12,
	"num_hidden_layers": 12,
	"type_vocab_size": 1,
	"vocab_size": 50265
}

with open("tswana_models/config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer_config = {"max_len": 512}

with open("tswana_models/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)

