from setuptools import setup

setup(
	name='qfb_env',
	version='1.0',
	description='QFBEnv Reinforcement Learning OpenAI Gym environment.',
	author='lgrech',
	euthor_email='leander.grech@cern.ch',
	url='https://gitlab.cern.ch/lgrech/eosbinreader',
	packages=['qfb_env'],
	install_requires=['numpy', 'gym']
)
