from setuptools import setup, find_packages

setup(
    name='VideoGameRec',
    version='1.0.0',
    description='A game recommendation system using KNN and Neural Network models',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/Bal67/VideoGameRec',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'scipy',
        'tensorflow',
        'streamlit',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'run_features=VideoGameRec.scripts.features:main',
            'run_models=VideoGameRec.scripts.models:main',
            'run_app=VideoGameRec.app:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

