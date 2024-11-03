import subprocess

from setuptools import setup, find_packages


def get_git_version():
    """Get version from git tags"""
    try:
        # Get the latest git tag
        version = subprocess.check_output(['git', 'describe', '--tags']).decode().strip()

        # If there are no tags, use 0.1.0-dev
        if not version:
            return '0.1.0-dev'

        # Clean the version string
        if '-' in version:
            # Format: v1.2.3-N-ghash -> 1.2.3.devN
            tag, commits, _ = version.split('-')
            return f"{tag.lstrip('v')}.dev{commits}"

        # Just return the tag without the 'v' prefix
        return version.lstrip('v')
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git is not available or not a git repository
        return '0.1.0-dev'

setup(
    name="buckshot-core",
    version=get_git_version(),
    packages=find_packages(),
    install_requires=[],
    author="sirily11",
    author_email="sirily1997@gmail.com",
    description="Core game logic for Buckshot game",
    long_description_content_type="text/markdown",
    url="https://github.com/sirily11/buckshot-core",
    classifiers=[],
    python_requires=">=3.10",
)