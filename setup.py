from setuptools import setup, find_packages

setup(
    name='ImageRetrieval',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1+cu117',
        'torchvision==0.14.1+cu117',
        'tqdm==4.64.1',
        'scipy==1.10.1',
        'sklearn==0.0.post1',
        'numpy==1.25.0',
        'opencv-python==4.7.0.72',
        'matplotlib==3.7.0',
        'Pillow==8.4.0'
    ],
    author='Morteza Mahdavi',
    description='This is my install packages setup for Image Retrieval project (RoshanAi Company internship)',
)