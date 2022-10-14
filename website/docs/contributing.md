---
sidebar_position: 7
---

# Developer Guide
Hi Developer! Great that you want to contribute to `fseval`. Let's get you started as soon as possible.

## Cloning `fseval`
First, clone the repository to your local computer. Do the following:

```
git clone https://github.com/dunnkers/fseval.git
```

> Make sure you have [Git](https://git-scm.com/) installed.

This creates a folder called `fseval`. Open it in your prefered editor.

## Installing the required packages

### Option A: using a Devcontainer
If you happen to use VSCode as your editor, you can open `fseval` in a [**Devcontainer**](https://code.visualstudio.com/docs/remote/containers). Devcontainers allow you to develop _inside_ a Docker container - which means all dependencies and packages are automatically set up for you. First, make sure you have the [Remote Development extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) installed.


Then, you can do two things.

1. Click the following button:

    [![Open in Remote - Containers](https://img.shields.io/static/v1?label=Remote%20-%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/dunnkers/fseval)

1. Or, open up the repo in VSCode. Then, you should see the following notification:

![reopen in devcontainer](/img/contributing/reopen-in-devcontainer.png)

Now you should have a fully working dev environment working 🙌🏻. You can run tests, debug code, etcetera. All dependencies are automatically installed for you.

Run/debug tests:

![tests view vscode](/img/contributing/tests-view-vscode.png)

🙌🏻

### Option B: installing the dependencies manually
Make sure you are in the `fseval` folder (cloned in the previous step), and run:

```
pip install -r requirements.txt
pip install -r .devcontainer/requirements.txt
pip install .
```

Now, you should be able to run tests:

```
pytest tests
```
![pytest started](/img/contributing/pytest-started.png)

## Developing the website
The documentation website is built with [Docusaurus](https://docusaurus.io/). To start the server, do the following:

```
cd website
yarn
yarn start
```

Which should start the documentation website:

![docs website started](/img/contributing/docs-website-started.png)

🙌🏻.

You can now edit everything in `docs` and the website should be live updated at [http://localhost:3000/fseval/](http://localhost:3000/fseval/). Create a Pull Request once you are done making edits. Cheers!