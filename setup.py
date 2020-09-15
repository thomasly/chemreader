from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="chemreader",
    version="0.3.4",
    author="Yang Liu",
    author_email="thomasliuy@gmail.com",
    description="Read data from typical chemical file formats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thomasly/chem_reader",
    download_url="https://github.com/thomasly/chem_reader/archive/0.3.3.tar.gz",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy"],
    python_requires=">=3.6",
)
