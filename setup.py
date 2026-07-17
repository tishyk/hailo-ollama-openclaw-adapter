"""Fallback installer for environments with older pip/setuptools.

Modern pip (>=21.3) with modern setuptools reads pyproject.toml directly and
does not need this file. Older tooling found in some shipped virtualenvs
(such as Hailo's DFC venv) falls back to setup.py. Metadata is duplicated
here so those older environments still get correct package info.

Keep pyproject.toml as the source of truth; update this file in lockstep
when bumping version.
"""

from setuptools import find_packages, setup


def _read_long_description() -> str:
    try:
        with open("README.md", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


setup(
    name="hailo-ollama-openclaw-adapter",
    version="2.0.0",
    description=(
        "FastAPI adapter bridging Hailo-Ollama to OpenClaw 2026.4.20 on Raspberry Pi"
    ),
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="Sergii Tishchenko",
    license="MIT",
    url="https://github.com/tishyk/hailo-ollama-openclaw-adapter",
    project_urls={
        "Homepage": "https://github.com/tishyk/hailo-ollama-openclaw-adapter",
        "Repository": "https://github.com/tishyk/hailo-ollama-openclaw-adapter",
        "Issues": (
            "https://github.com/tishyk/hailo-ollama-openclaw-adapter/issues"
        ),
    },
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fastapi>=0.100",
        "httpx>=0.25",
        "uvicorn[standard]>=0.24",
    ],
    extras_require={
        "dev": [
            "ruff>=0.5",
            "pytest>=7",
            "pytest-asyncio>=0.23",
        ],
    },
    entry_points={
        "console_scripts": [
            "hailo-ollama-adapter=hailo_ollama_adapter.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: FastAPI",
    ],
    keywords=[
        "hailo",
        "ollama",
        "openclaw",
        "llm",
        "raspberry-pi",
        "ai-accelerator",
    ],
)
