# Madlibs: A Framework for Accelerated Structured Output Generation

Varun Shenoy & Alex Derhacobian

![A diagram](figs/diagram.png)

**Madlibs** is a Python framework that enables the efficient creation of structured output from an LLM by breaking up a target schema into atomic components and then performing generations in parallel.

Compared to a naive JSON generation pipeline relying on prompting and HF Transformers, we find Madlibs can generate outputs as much as **30x faster** on a custom dataset we curated.

## How does it work?

Structured output formats, such as JSON or YAML, have an inherent parallel or hierarchichal structure.

Consider the following unstructured passage:

```
Welcome to 123 Azure Lane, a stunning San Francisco residence boasting fantastic contemporary design, now on the market for $2,500,000. Spread out over a luxurious 3,000 square feet, this property combines sophistication and comfort to create a truly unique living experience.

An idyllic home for families or professionals, our exclusive residence is equipped with five spacious bedrooms, each oozing warmth and modern elegance. The bedrooms are carefully planned to allow ample natural light and generous storage space. With three elegantly designed full bathrooms, the residence guarantees convenience and privacy for its residents.

The grand entrance leads you to a spacious living area, providing an excellent ambience for gatherings or a quiet evening by the fire. The chef's kitchen includes state-of-the-art appliances, custom cabinetry, and beautiful granite countertops making it a dream for anyone who loves to cook.

Enjoy the beauty of San Francisco through large windows that provide not only an abundance of sunlight but spectacular city views. This gem also includes a beautifully landscaped patio, perfect for outdoor family gatherings or silent reflection in the lap of nature.

Located in one of San Francisco’s most coveted neighborhoods, 123 Azure Lane seamlessly marries gorgeous architecture with modern comforts, making it indeed a dream dwelling worth investing in. Don't miss the chance to make this house your forever home.
```

If we want to extract `address`, `square footage`, `number of bedrooms`, `number of bathrooms`, and `price` using an LLM, we could ask the model to fill in a schema according to the description.

```
{
  "address": "123 Azure Lane",
  "price": 2500000,
  "square_feet": 3000,
  "num_beds": 5,
  "num_baths": 3
}
```

This is how most teams currently extract structured output from unstructured text using LLMs.

However, this is inefficient.

Notice how each of these keys are independent of one another. Madlibs takes advantage of **prompt parallelism** by treating every key-value pair in the schema as a separate inquiry.

For example, we can extract the `num_baths` without having already generated the `address`!

Moreover, LLMs are embarrasingly parallel and running queries in batches is much faster than in a serial order.

Thus, we can split up the schema over multiple queries. The LLM will then fill in the schema for each independent key **in parallel** and emit far fewer tokens in a single pass, allowing for much faster inference times.

## Installation

## Examples

## Testing

## Citation
