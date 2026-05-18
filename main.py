from fastapi import FastAPI
from search import search_web
# from transformers import pipeline
# import summarize as sz
import decision as dcsn
import planner as pln
import re

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# adding agent memory
memory = []

MAX_STEPS = 2
MEMORY_SIZE = 5
MODEL_NAME = "google/flan-t5-small"

# generator = pipeline("text2text-generation", model=MODEL_NAME)

# for Production API calls
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def is_math_expression(query: str) -> bool:
    # Only allow digits, operators, parentheses, spaces
    if not re.fullmatch(r'[0-9+\-*/().\s]+', query):
        return False

    # Must contain at least one operator
    if not any(op in query for op in ["+", "-", "*", "/"]):
        return False

    return True

def format_memory(memory):
    formatted = ""
    for m in memory:
        formatted += f"User: {m['query']}\nAssistant: {m['answer']}\n"
    return formatted

def get_base_context(memory):
    followup_words = [
        "example",
        "simplify",
        "simple",
        "summarize",
        "explain it",
        "in simple terms",
        "that",
        "it"
    ]

    # search backwards through memory
    for m in reversed(memory):
        query_lower = m["query"].lower()

        # skip follow-up style queries
        if not any(word in query_lower for word in followup_words):
            return m["answer"]

    return ""

def is_followup(query: str):
    return any(word in query.lower() for word in [
        "explain", "simplify", "summarize", "in simple terms","example", "it", "that"
    ])

def detect_intent(query: str):
    q = query.lower()

    if ("example" in q or "for instance" in q):
        return "example"
    elif ("simple" in q or "simplify" in q or "simply" or "plain word" in q):
        return "simplify"
    elif "summarize" in q:
        return "summarize"
    else:
        return "general"

# def generate_text(prompt: str):
#     result = generator(
#         prompt,
#         max_length=120,
#         do_sample=False,   # deterministic
#         repetition_penalty=1.5  # reduces looping
#     )[0]["generated_text"]
#     return result

def generate_text(prompt: str):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=200,
        frequency_penalty=0.5
    )
    return completion.choices[0].message.content

# main function
@app.get("/research")
def research(query: str):
    query = query.strip()

    observation = ""
    route = None
    resp = None

    for step in range(MAX_STEPS):
        # print(f"--- Step {step+1} ---")
        logger.info(f"--- Step={step+1} ---")

        # Rule overrides FIRST
        if is_math_expression(query):
            try:
                #print("tool: math")
                logger.info(f"tool=math")
                # NOTE:
                # restricted via regex to numeric math expressions only
                resp = str(eval(query.replace(" ", "")))
                route = "math"

                # logging
                logger.info(f"query={query}")
                logger.info(f"route={route}")
                logger.info(f"response_length={len(resp)}")
                break
            except:
                pass

        if is_followup(query):
            action = "answer"
        else:
            # action = pln.plan_step(query, observation)
            try:
                action = pln.plan_step(query, observation)
            except Exception as e:
                logger.error(f"search_failed: {str(e)}")
                return {
                    "status": "error",
                    "message": "Planner failed"
                }

        #print("plan:", action)
        logger.info(f"plan={action}")

        if action == "search":
            # print("tool: search")
            logger.info(f"tool=search")
            #text_dict = search_web(query)
            try:
                text_dict = search_web(query)
            except Exception as e:
                logger.error(f"search_failed: {str(e)}")
                return {
                    "status": "error",
                    "message": "Search failed"
                }
            observation = text_dict["answer"]

        elif action == "answer":
            # we get the intent based on the query first
            ans_type = detect_intent(query)
            # print("intent: ", ans_type)
            logger.info(f"intent={ans_type}")
            # print("tool: answer (LLM)")
            logger.info(f"tool=answer (LLM)")

            if ans_type in ["example", "simplify", "summarize"]:
                context = get_base_context(memory)
            else:
                context = format_memory(memory)

            # based on the intent type, we modify our prompts
            if ans_type == 'example':
                prompt = f"""
                Use the previous conversation as the PRIMARY source.
                Only use the additional information if needed.
                
                Previous conversation:
                {context}
                
                Additional information:
                {observation}
                
                User query:
                {query}
                
                Task:
                Give one simple real-world example.
                
                IMPORTANT:
                - Do NOT repeat the example given below
                - Create a DIFFERENT example
                
                Example:
                Explanation: Machine learning learns patterns from data.
                Example: A spam filter learns from past emails to classify new emails.
                
                Now generate a NEW example.
                
                Answer:
                """
            elif ans_type == 'simplify':
                prompt = f"""
                Use the previous conversation as the PRIMARY source.
                Only use the additional information if needed.
                
                Previous conversation:
                {context}
                
                Additional information:
                {observation}
                
                User query:
                {query}
                
                Task:
                Explain this in very simple terms, like teaching a beginner.
                Avoid technical jargon.
                
                Answer:
                """
            else:
                prompt = f"""
                Previous conversation:
                {context}
                
                Observation:
                {observation}
                
                User query:
                {query}
                
                Answer clearly:
                """
            #resp = generate_text(prompt)
            try:
                resp = generate_text(prompt)
            except Exception as e:
                logger.error(f"llm_failed: {str(e)}")
                resp = "Sorry, I couldn't generate a response."
            route = "answer" if not is_math_expression(query) else "math"
            logger.info(f"query={query}")
            logger.info(f"route={route}")
            logger.info(f"response_length={len(resp)}")
            break

    if resp is None:
        # print("fallback: search")
        logger.info(f"fallback=search")
        # text_dict = search_web(query)
        
        try:
            text_dict = search_web(query)
        except Exception as e:
            logger.error(f"search_failed: {str(e)}")
            return {
                "status": "error",
                "message": "Search failed"
            }

        prompt = f"""
        Search results:
        {text_dict["answer"]}
        
        User query:
        {query}
        
        Answer clearly:
        """

        #resp = generate_text(prompt)
        try:
            resp = generate_text(prompt)
        except Exception as e:
            logger.error(f"llm_failed: {str(e)}")
            resp = "Sorry, I couldn't generate a response."
        route = "search"
        logger.info(f"query={query}")
        logger.info(f"route={route}")
        logger.info(f"response_length={len(resp)}")

    # storing memory
    memory.append({
        "query": query,
        "route": route,
        "answer": resp
    })
    # keep only last 5
    memory[:] = memory[-MEMORY_SIZE:]

    return {
        "status": "success",
        "query": query,
        "route": route,
        "answer": resp,
        "steps": step + 1
    }
