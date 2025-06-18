# G11FinalProject
Final Project for Group 11 of the AI for Software Developers course.

## Requirements
- Python 3.7 or higher
- HuggingFace Access Token
- OpenAI API Key

## Installation
1. Set up virtual environment

From top-level project directory, run the following:

`python3 -m venv .finalprojvenv`

Then, activate the environment, with a different command depending on OS:

- Unix: (source .finalprojvenv/bin/activate )
- Powershell: (.\finalprojvenv\Scripts\Activate.ps1)
- Windows: (finalprojvenv\Scripts\activate.bat)

Finally, install the required python packages:

`pip install -r requirements.txt`

2. Set up environment variables

Rename `example.env` to `.env`, and populate the `HF_TOKEN` and `OPENAI_API_KEY` variables with your HuggingFace Token and OpenAI API Key respectively.

## Usage
To run locally & through Gradio's hosting, simply run `python3 src/hf_gradio_ai_app.py`. This will start both a local and Gradio-hosted instance, with the URL printed in the terminal.

To deploy to HuggingFace, `cd src` and then run `gradio deploy`.
