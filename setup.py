from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="semantic_splitter",
    version="0.1.0",
    description="Semantic chunking of documents using Sentence Transformers and LangChain, without relying on fixed chunk sizes or overlaps.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Maran M",
    author_email="mahemaran99@gmail.com",
    url="https://github.com/Mahemaran/semantic_splitter",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "sentence-transformers",
        "langchain",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
)
