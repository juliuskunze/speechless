from distutils.core import setup

from pip.req import parse_requirements

requirements = [str(ir.req) for ir in parse_requirements("requirements.txt", session=False)]

setup(
    name='speechless',
    version='0.1',
    packages=['speechless'],
    url='https://github.com/JuliusKunze/speechless',
    license='MIT License',
    author='Julius Kunze',
    author_email='juliuskunze@gmail.com',
    description='',
    install_requires=requirements
)
