from setuptools import setup, find_packages

setup(
    name='clipcat',
    version='0.1',
    py_modules=['clipcat_tool'],
    install_requires=[
        'gymnasium==0.29.1',
        'numpy==1.26.4',
        'Pillow==10.3.0',
        'transformers==4.41.2',
        'torch==2.3.1',
        'stable-baselines3==2.3.2',
        'virtualenv==20.26.2'
    ],
    entry_points={
        'console_scripts': [
            'clipcat=clipcat_tool:main',
        ],
    },
    python_requires='==3.11.7',
)
