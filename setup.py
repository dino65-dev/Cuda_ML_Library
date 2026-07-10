import os
import sys
from setuptools import setup, find_packages, Extension
import numpy as np

# Read version from file
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    return '0.1.0'

# Read README for long description
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Check if building CPU-only version
CPU_ONLY = os.environ.get('CUDA_ML_CPU_ONLY', '0') == '1'

# Package requirements
install_requires = [
    'numpy>=1.19.0',
    'scikit-learn>=1.0.0',
]

# Optional CUDA requirements (for documentation)
extras_require = {
    'cuda': [
        'cupy-cuda12x>=12.0.0',  # For CUDA 12.x support
    ],
    'dev': [
        'pytest>=6.0',
        'pytest-cov',
        'black',
        'flake8',
        'sphinx',
        'sphinx-rtd-theme',
    ],
    'examples': [
        'matplotlib>=3.0.0',
        'seaborn>=0.11.0',
        'jupyter>=1.0.0',
    ],
    'dspark': [
        'torch>=2.1.0',
    ],
}

setup(
    name='cuda-ml-library',
    version=get_version(),
    author='dino65-dev',
    author_email='your-email@example.com',  # Add your email
    description='High-performance CUDA-accelerated Machine Learning library',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/dino65-dev/Cuda-ML-Library',
    project_urls={
        'Bug Reports': 'https://github.com/dino65-dev/Cuda-ML-Library/issues',
        'Source': 'https://github.com/dino65-dev/Cuda-ML-Library',
        'Documentation': 'https://github.com/dino65-dev/Cuda-ML-Library#readme',
    },
    packages=find_packages(),
    package_data={
        'SVM': ['*.py'],
        'HBM_SVM': ['*.py'],
        'DSpark': ['*.py', 'README.md'],
        'Usage': ['**/*.py', '**/*.md'],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='cuda, gpu, machine learning, svm, support vector machine, high performance computing',
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'cuda-ml-info=SVM.cuda_svm:print_info',
        ],
    },
    zip_safe=False,  # Due to shared libraries
)
