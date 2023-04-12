import setuptools

setuptools.setup(
    name="Roundtrip", 
    version="2.0.0",
    author="Qiao Liu",
    author_email="liuqiao@stanford.edu",
    description="Roundtrip: density estimation with deep generative neural networks",
    long_description="Density estimation is one of the fundamental problems in both statistics and machine learning. In this study, we propose Roundtrip, a computational framework for general-purpose density estimation based on deep generative neural networks. Roundtrip retains the generative power of deep generative models, such as generative adversarial networks (GANs) while it also provides estimates of density values, thus supporting both data generation and density estimation. Unlike previous neural density estimators that put stringent conditions on the transformation from the latent space to the data space, Roundtrip enables the use of much more general mappings where target density is modeled by learning a manifold induced from a base density (e.g., Gaussian distribution). Roundtrip provides a statistical framework for GAN models where an explicit evaluation of density values is feasible. In numerical experiments, Roundtrip exceeds state-of-the-art performance in a diverse range of density estimation tasks. Roundtrip is freely available at https://github.com/kimmo1019/Roundtrip.",
    long_description_content_type="text/markdown",
    url="https://github.com/kimmo1019/Roundtrip",
    packages=setuptools.find_packages(),
    install_requires=[
   'tensorflow>=2.8.0',
   'scikit-learn',
   'pandas',
   'python-dateutil'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
    'console_scripts': [
        'roundtrip = Roundtrip.cli:main',
    ]},
)