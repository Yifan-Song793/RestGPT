# RestGPT

RestGPT: An LLM-Based Autonomous Agent Controlling Real-World Applications via RESTful APIs

The data of our proposed RestBench is in `datasets` folder.

## Setup

```bash
pip install langchain colorama tiktoken spotipy openai
```

create `logs` folder

Get OpenAI key from OpenAI, TMDB key from https://developer.themoviedb.org/docs/getting-started, and Spotify key from https://developer.spotify.com/documentation/web-api

Fill in your own key in `config.yaml`

## (Optional) Initialize the Spotify Environment

**WARNING: this will remove all your data from spotify!**

```python
python init_spotify.py
```

## Run

```bash
# TMDB
python run_tmdb.py

# Spotify，please open the Spotify on your device
python run_spotify.py
```

The log file will be in the `logs` folder
