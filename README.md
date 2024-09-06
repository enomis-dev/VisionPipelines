# VisionPipelines

<div style="text-align: center;">
    <img src="project_image.webp" alt="VisionPipeline Thumbnail" width="400"/>
</div>

The purpose of this project is to easy the use of advanced image processing algorithms for most common tasks as Image registration, segmentation, object detection and others...



## Contributing to VisionPipelines

Thank you for your interest in contributing to VisionPipeline! To ensure a smooth development experience and maintain consistency across contributions, please follow the guidelines below.

### Getting Started

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/visionpipeline.git
   cd visionpipeline
   ```

2. **Set Up Your Development Environment**

   VisionPipeline uses [Poetry](https://python-poetry.org/) for dependency management. Poetry simplifies the process of installing and managing dependencies.

   - **Install Poetry** (if you haven't already):

     ```bash
     pip install poetry
     ```

   - **Create a Virtual Environment** (optional):

     If you prefer to use a virtual environment manually rather than Poetry’s built-in environment, you can create one with:

     ```bash
     python -m venv venv
     ```

     Then activate the virtual environment:

     - **On Windows**:
       ```bash
       venv\Scripts\activate
       ```

     - **On macOS and Linux**:
       ```bash
       source venv/bin/activate
       ```

   - **Install Project Dependencies**:

     Using Poetry:

     ```bash
     poetry install
     ```

### Development and Testing

- **Run Tests**: Ensure all tests pass before submitting a contribution. Run the test suite with:

  ```bash
  pytest
  ```

- **Code Style**: Follow the [PEP 8](https://pep8.org/) style guide for Python code. Consistent formatting helps maintain readability and quality.

### Making a Contribution

1. **Fork the Repository**

   Create a fork of the repository on GitHub to make your changes.

2. **Create a Feature Branch**

   Create a new branch for your changes:

   ```bash
   git checkout -b your-feature-branch
   ```

3. **Make Your Changes**

   Edit code, add features, or fix bugs as necessary. Commit your changes with a descriptive message:

   ```bash
   git add .
   git commit -m "Describe your changes"
   ```

4. **Push Your Changes**

   Push your branch to your forked repository:

   ```bash
   git push origin your-feature-branch
   ```

5. **Submit a Pull Request**

   Open a pull request (PR) on GitHub from your forked repository to the original repository. Provide a clear description of your changes and why they should be merged.

### Reviewing and Merging

- **Code Review**: All pull requests will be reviewed by the maintainers. Feedback will be provided, and necessary changes may be requested.

- **Merge**: Once your pull request is approved and passes all tests, it will be merged into the main branch.

### Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [PEP 8 Style Guide](https://pep8.org/)

Thank you for contributing to VisionPipeline!


### Notebooks

In order to play with the notebooks, if you have to create a local env, follow the section "Getting Started".

1. **Install jupyter**

```bash
pip install jupyter ipykernel
```

2. **Add kernel to jupyter notebook**
python -m ipykernel install --user --name your_env_name --display-name "your_env_name"

3. **Start jupyter-notebook**
```bash
jupyter notebook
```
