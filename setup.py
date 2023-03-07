import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="rllbm",
    version="0.0.1",
    author="Hugues de Laroussilher",
    author_email="huguesdelaroussilhe@gmail.com",
    description="A playground for training RL agents to interact with fluid simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hlasco/rllbm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    packages=setuptools.find_namespace_packages(include = ["rllbm", "rllbm.*"]),
    python_requires=">=3.8",
)