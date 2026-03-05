# New Project Setup

How to use this template


Click “Use this template”

Clone your new repo

Create env from environment.yml

Select interpreter + kernel in VS Code

Put raw data outside git (or in ignored data/)

Start work in src/ + notebooks/ + scripts/

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Or information from setting up a project from scratch

1.1 Create project directory
mkdir -p ~/projects/project-name
cd ~/projects/project-name
1.2 Initialize Git repository
git init

Create initial files:

README.md
.gitignore

Example .gitignore entries:

data/
outputs/
*.tif
*.zarr
.ipynb_checkpoints/
__pycache__/
.DS_Store

Commit initial version:

git add -A
git commit -m "Initial commit"
1.3 Create GitHub repository

Create a repo on GitHub and connect it:

git remote add origin https://github.com/<username>/<repo>.git
git branch -M main
git push -u origin main
2. Create the Conda Environment

Create a new environment for the project.

Example:

conda create -n project-env python=3.11
conda activate project-env

Install core scientific packages:

conda install -c conda-forge numpy pandas matplotlib scikit-image
conda install -c conda-forge jupyterlab ipykernel

Install additional packages as needed:

conda install -c conda-forge napari
pip install stardist

Prefer conda-forge packages first, then pip if necessary.

3. Save the Environment (environment.yml)

The environment.yml file records the packages needed to recreate the environment.

It is not automatically updated.

Export environment periodically:

conda env export --from-history > environment.yml

Commit it:

git add environment.yml
git commit -m "Update environment"
git push

Another machine can recreate the environment with:

conda env create -f environment.yml
4. Open the Project in VS Code

Open the project folder:

File → Open Folder

Install extensions:

Python (Microsoft)
Jupyter (Microsoft)
5. Select the Python Interpreter

The interpreter is the Python executable VS Code uses for .py files.

Open the Command Palette:

Python: Select Interpreter

Select:

Python (project-env)

You should see the environment name in the VS Code status bar once a Python file is open.

6. Configure Notebooks
6.1 What is a Notebook Kernel?

A kernel is the Python environment used by a notebook.

Interpreter → used by .py files
Kernel → used by .ipynb notebooks

They should normally be the same environment.

6.2 Select Notebook Kernel

Open a notebook:

notebooks/example.ipynb

Top-right:

Select Kernel

Choose:

Python Environments → project-env

VS Code will remember this for the notebook.

6.3 Verify the Kernel

Run:

import sys
print(sys.executable)

The path should include the environment name:

.../envs/project-env/bin/python

If it shows base, the wrong environment is being used.

6.4 If the Environment Does Not Appear

Register the environment as a kernel:

conda activate project-env
python -m pip install ipykernel
python -m ipykernel install --user --name project-env --display-name "Python (project-env)"

Restart VS Code afterward.

7. Configure Notebook Working Directory

Notebooks often start in the notebooks/ directory, which breaks imports.

Create:

.vscode/settings.json

Add:

{
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}

This forces notebooks to run relative to the repository root.

Verify:

from pathlib import Path
Path.cwd()

It should return the project root.

Commit this configuration.

8. Recommended Project Structure
repo_root/

  src/
    project_package/
      __init__.py
      utils.py
      io.py
      segmentation.py
      tracking.py

  notebooks/
    01_exploration.ipynb
    02_algorithm_tests.ipynb
    03_analysis.ipynb

  scripts/
    run_segmentation.py
    run_tracking.py

  environment.yml
  README.md
  .gitignore
  .vscode/settings.json
9. Rules for Code Organization
.py files (src/)

Contain reusable code:

segmentation algorithms

image processing functions

data loading utilities

tracking logic

Example:

src/project_package/segmentation.py
Notebooks

Used for:

exploration

visualization

testing algorithms

documenting results

Avoid placing reusable logic directly in notebooks.

10. Daily Workflow

Start work:

git pull
conda activate project-env

Work in VS Code (edit code, run notebooks).

Commit changes regularly:

git add -A
git commit -m "Describe change"
git push
11. Moving Between Computers
On the second computer

Clone the repository:

git clone <repo-url>
cd repo-name

Create environment:

conda env create -f environment.yml
conda activate project-env

Open the project in VS Code.

Select interpreter and kernel.

Everything should now match the original environment.

12. Environment Management Guidelines

Environment contains:

Python
Scientific libraries
Visualization tools
Machine learning frameworks

The environment.yml file contains the instructions needed to recreate that environment.

Update it whenever packages change:

conda env export --from-history > environment.yml

Commit the update.