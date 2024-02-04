from setuptools import setup, find_packages

setup(
    name="madlibs",
    version='0.1.0',
    author="Varun Shenoy",
    author_email="vnshenoy@stanford.edu",
    description='A framework for faster structured output generation using LLMs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/varunshenoy/madlibs',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "prettytable==3.9.0",
        "pydantic==2.5.2",
        "torch",
        "tqdm",
        "vllm==0.3.0",
        "transformers==4.37.2",
    ]
)