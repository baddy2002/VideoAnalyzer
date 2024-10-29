from setuptools import setup, find_packages


# Funzione per leggere il file requirements.txt
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()
    
setup(
    name='video-analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),  # Ottieni dipendenze dinamicamente
    entry_points={
        'console_scripts': [
            'run-video-analysis=app.main:main',
        ],
    },
)
