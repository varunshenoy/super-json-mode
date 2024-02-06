from setuptools import setup, find_packages

setup(
    name="super-json-mode",
    version="0.1.2",
    author="Varun Shenoy",
    author_email="vnshenoy@stanford.edu",
    description="A framework for faster structured output generation using LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/varunshenoy/super-json-mode",
    packages=find_packages(),
    install_requires=[
        "prettytable==3.9.0",
        "pydantic==2.5.2",
        "tqdm",
        # "vllm==0.3.0",
        "transformers==4.37.2",
        "openai==1.11.1",
    ],
)
