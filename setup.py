from setuptools import setup

setup(
    name='pgs_recon',
    version='1.0.0',
    description='Photogrammetry reconstruction pipeline using OpenMVG + OpenMVS.',
    url='https://code.vis.uky.edu/seales-research/photogrammetry',
    author='University of Kentucky',
    license='MS-RSL',
    packages=['pgs_recon'],
    install_requires=[
        'numpy',
        'pyexiftool',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'pgs-recon = pgs_recon.reconstruct:main',
            'pgs-import = pgs_recon.sfm:main',
        ],
    },
    zip_safe=False,
)
