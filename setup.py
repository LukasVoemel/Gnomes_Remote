import setuptools

setuptools.setup(
    name="submission",
    version="0.0.0b0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'dragg',
        'dragg-comp',
        'redis'
        'tensorflow'
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9"
)
