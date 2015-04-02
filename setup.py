from setuptools import setup, find_packages

setup(name='Panoptisim',
      version='0.1',
      description='Hybrid Deployment Simulator',
      author='Dan Levin',
      author_email='dlevin@net.t-labs.tu-berlin.de',
      url='https://github.com/badpacket/sdn-deployment-problem',
      packages=['sim', 'traces'],
      scripts=['panoptisim.py'],
      install_requires=['networkx>=1.7', 'matplotlib>=1.1'],
      package_data = {'pickles': ['*.pickle'],
                      'traces': ['*.json']},
     )
