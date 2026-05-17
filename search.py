from fastapi import FastAPI, HTTPException
import requests
DDG_BASE_URL = "https://api.duckduckgo.com/"

def search_web(query: str):
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }

    response = requests.get('https://api.duckduckgo.com/', params=params)
    data = response.json()
    if data.get("AbstractText"):
        cleaned_answer = data.get("AbstractText")
    elif data.get("Definition"):
        cleaned_answer = data.get("Definition")
    else:
        cleaned_answer = "search did not yeild any results"

    return {
    	"query": query,
    	"answer": cleaned_answer
    }