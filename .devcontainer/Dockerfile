FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

RUN /usr/local/bin/python -m pip install --upgrade pip

# Install Node>=16.x which required for Docusaurus (the documentation website)
RUN sudo apt update -y && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - && \
    sudo apt install nodejs -y