from distutils.sysconfig import get_python_inc

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def get_lines(relative_path):
    with open(relative_path) as f:
        return f.readlines()


def get_version(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split('"')[1]
    raise RuntimeError("Unable to find version string.")


COMPILER_DIRECTIVES = {
    "language_level": "3",
}
MOD_NAMES = ["edsnlp.matchers.phrase"]

include_dirs = [
    numpy.get_include(),
    get_python_inc(plat_specific=True),
]
ext_modules = []
for name in MOD_NAMES:
    mod_path = name.replace(".", "/") + ".pyx"
    ext = Extension(
        name,
        [mod_path],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++11"],
    )
    ext_modules.append(ext)
print("Cythonizing sources")
ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

factories = [
    "matcher = edsnlp.components:matcher",
    "advanced = edsnlp.components:advanced",
    "endlines = edsnlp.components:endlines",
    "sentences = edsnlp.components:sentences",
    "normalizer = edsnlp.components:normalizer",
    "accents = edsnlp.components:accents",
    "lowercase = edsnlp.components:remove_lowercase",
    "pollution = edsnlp.components:pollution",
    "quotes = edsnlp.components:quotes",
    "charlson = edsnlp.components:charlson",
    "sofa = edsnlp.components:sofa",
    "priority = edsnlp.components:priority",
    "ccmu = edsnlp.components:ccmu",
    'gemsa" = edsnlp.components:gemsa',
    'covid" = edsnlp.components:covid',
    "history = edsnlp.components:history",
    "family = edsnlp.components:family",
    "hypothesis = edsnlp.components:hypothesis",
    "negation = edsnlp.components:negation",
    "rspeech = edsnlp.components:rspeech",
    "consultation_dates = edsnlp.components:consultation_dates",
    "dates = edsnlp.components:dates",
    "reason = edsnlp.components:reason",
    "sections = edsnlp.components:sections",
    "context = edsnlp.components:context",
    "measures = edsnlp.components:measures",
    "pseudonymisation = edsnlp.components:pseudonymisation",
]

setup(
    name="edsnlp",
    version=get_version("edsnlp/__init__.py"),
    author="Data Science - DSI APHP",
    author_email="basile.dura-ext@aphp.fr",
    description=(
        "A set of spaCy components to extract information "
        "from clinical notes written in French."
    ),
    url="https://github.com/aphp/edsnlp",
    project_urls={
        "Documentation": "https://aphp.github.io/edsnlp",
        "Demo": "https://aphp.github.io/edsnlp/demo",
        "Bug Tracker": "https://github.com/aphp/edsnlp/issues",
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=get_lines("requirements.txt"),
    ext_modules=ext_modules,
    extras_require=dict(
        demo=["streamlit>=1.2"],
        distributed=["pyspark"],
    ),
    package_data={
        "edsnlp": ["resources/*"],
        "": ["*.pyx", "*.pxd", "*.pxi"],
    },
    entry_points={
        "spacy_factories": factories,
        "spacy_languages": ["eds = edsnlp.language:EDSLanguage"],
    },
)
