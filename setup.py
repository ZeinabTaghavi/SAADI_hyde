from setuptools import setup, find_packages


setup(
    name='hyde',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'cohere',
        'hf_xet',
        'huggingface_hub>=0.34.0',
        'openai',
        'pyserini',
        'faiss-cpu',
        'safetensors',
        'transformers>=4.51.0',
        'accelerate'
    ],
)
