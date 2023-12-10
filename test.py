import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from madlibs.integrations.transformers import StructuredOutputForModel
from pydantic import BaseModel


class Query(BaseModel):
    title: str
    authors: str
    summary: str
    core_contribution: str


device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model and tokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# Create a structured output object
structured_model = StructuredOutputForModel(model, tokenizer)

# Generate the output
prompt = """Mamba: Linear-Time Sequence Modeling with Selective State Spaces
Albert Gu, Tri Dao
Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation."""

print(structured_model.generate(prompt, schema=Query, batch_size=4))

# prompt = "Pi is 3.1. Repeat Pi: "
# output = structured_output.generate

# class Ingredients(BaseModel):
#     """Data model representing a single ingredient."""

#     name: str
#     amount: float
#     unit: str


# class MenuItem(BaseModel):
#     """Data model representing a single menu item."""

#     name: str
#     price: float
#     description: str
#     calories: int
#     main_ingredient: Ingredients


# # print(data)

# schema = SchemaBatcher(MenuItem, batch_size=2)
# print(schema.batches)

# m = MenuItem(
#     name="Cheeseburger",
#     price=5.99,
#     description="A classic cheeseburger",
#     calories=750,
#     main_ingredient=Ingredients(name="Beef", amount=0.5, unit="lb"),
# )

# print(m.schema()["properties"])
