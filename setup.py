from setuptools import setup

setup(name='sc2bot',
      version='0.1',
      description='Starcraft II RL Bot',
      url='https://github.com/alanxzhou/sc2bot',
      author='Alan Zhou',
      author_email='alan.xiang.zhou@gmail.com',
      license='General Public License',
      packages=['sc2bot'],
      zip_safe=False, install_requires=['pysc2', 'numpy', 'matplotlib', 'torch'])