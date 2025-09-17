"""
Setup script per il sistema RAG avanzato
"""
from setuptools import setup, find_packages
from pathlib import Path

# Leggi il README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Dipendenze base
install_requires = [
    "langchain>=0.1.0",
    "langchain-openai>=0.1.0",
    "langchain-experimental>=0.0.50",
    "langchain-community>=0.0.20",
    "PyPDF2>=3.0.0",
    "faiss-cpu>=1.7.4",
    "openai>=1.0.0",
    "rank-bm25>=0.2.2",
    "google-generativeai>=0.3.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]

# Dipendenze opzionali
extras_require = {
    "docling": ["docling>=1.0.0"],
    "advanced": [
        "nltk>=3.8",
        "spacy>=3.6.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "loguru>=0.7.0",
    ],
}

# All extras
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="advanced-rag-system",
    version="1.0.0",
    author="RAG Team",
    author_email="",
    description="Sistema RAG avanzato con Docling, Semantic Chunking, HyPE e Fusion Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "rag-system=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
