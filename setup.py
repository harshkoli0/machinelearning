from setuptools import setup, find_packages
from typing import List

hypen="-e ."

def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements    



setup(
    name="mlpro",
    version="0.0.1",
    author="harsh",
    author_email="hkoli3319@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
