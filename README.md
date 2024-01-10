# Madlibs: A Framework for Accelerated Structured Output Generation

![A diagram](figs/diagram.png)

**Madlibs** is a Python framework that enables the efficient creation of structured output from an LLM by breaking up a target schema into atomic components and then performing generations in parallel.

Compared to a naive JSON generation pipeline relying on prompting and HF Transformers, we find Madlibs can generate outputs as much as **30x faster** on a custom dataset we curated.

Built by Varun Shenoy & Alex Derhacobian

## How does it work?

Structured output formats, such as JSON or YAML, have an inherent parallel or hierarchichal structure.

Consider the following unstructured passage:

> Welcome to 123 Azure Lane, a stunning San Francisco residence boasting fantastic contemporary design, now on the market for $2,500,000. Spread out over a luxurious 3,000 square feet, this property combines sophistication and comfort to create a truly unique living experience.

> An idyllic home for families or professionals, our exclusive residence is equipped with five spacious bedrooms, each oozing warmth and modern elegance. The bedrooms are carefully planned to allow ample natural light and generous storage space. With three elegantly designed full bathrooms, the residence guarantees convenience and privacy for its residents.

> The grand entrance leads you to a spacious living area, providing an excellent ambience for gatherings or a quiet evening by the fire. The chef's kitchen includes state-of-the-art appliances, custom cabinetry, and beautiful granite countertops making it a dream for anyone who loves to cook.

If we want to extract `address`, `square footage`, `number of bedrooms`, `number of bathrooms`, and `price` using an LLM, we could ask the model to fill in a schema according to the description.

A potential schema (such as one generated from a Pydantic object) could look like this:

```
{
    "address": {
        "type": "string"
    },
    "price": {
        "type": "number"
    },
    "square_feet": {
        "type": "integer"
    },
    "num_beds": {
        "type": "integer"
    },
    "num_baths": {
        "type": "integer"
    }
}
```

And a valid output could look something like this:

```
{
  "address": "123 Azure Lane",
  "price": 2500000,
  "square_feet": 3000,
  "num_beds": 5,
  "num_baths": 3
}
```

This is currently how most teams currently extract structured output from unstructured text using LLMs.

However, this is inefficient.

Notice how each of these keys are independent of one another. Madlibs takes advantage of **prompt parallelism** by treating every key-value pair in the schema as a separate inquiry.

For example, we can extract the `num_baths` without having already generated the `address`!

Moreover, LLMs are embarrasingly parallel and running queries in batches is much faster than in a serial order.

Thus, we can split up the schema over multiple queries. The LLM will then fill in the schema for each independent key **in parallel** and emit far fewer tokens in a single pass, allowing for much faster inference times.

## Installation

1. Create a conda environment and install the dependencies

```
conda create --name madlibs python=3.10 -y
conda activate madlibs
```

2. Clone and install the dependencies

```
git clone https://github.com/varunshenoy/madlibs
cd madlibs
pip install -r requirements.txt
```

## Examples

...

## Roadmap

There's a lot of features that can make Madlibs better, but we're just two guys with a limited amount of time. Would love any help!

-[ ] Structured sampling: ideally, we should mask the LLM's logits to enforce type constraints, similar to JSONFormer. There are a few packages out there that already do this, and either those should integrate our parallelized JSON generation pipeline or we should build it out into Madlibs.

-[ ] Llama.cpp Support: Madlibs works best in local situations where batch size is generally 1. You can exploit batchingto reduce latency, similar to speculative decoding. Llama.cpp is the premier framework for local models + cpu inference.

-[ ] TRT-LLM Support: vLLM is great and easy to use, but ideally we integrate with a much more performant framework like TRT-LLM.

## Citation

We appreciate it if you would please cite the following paper if you found the library useful for your work:

```
@article{madlibs2024,
  title={Madlibs: A Framework for Accelerated Structured Output Generation},
  author={Shenoy, Varun and Derhacobian, Alex},
  journal={arXiv},
  year={2024}
}
@misc{ShenoyDerhacobian2024,
  author = {Shenoy, Varun and Derhacobian, Alex},
  title = {Madlibs: A Framework for Accelerated Structured Output Generation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/varunshenoy/madlibs}}
}
```

_This project was built for CS 229: Systems for Machine Learning. Huge thanks to the teaching team and TAs for their guidance throughout this project._
