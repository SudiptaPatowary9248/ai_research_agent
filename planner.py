from transformers import pipeline

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

planner = pipeline("text2text-generation", model="google/flan-t5-small")

# for multi-step reasoning
observation = ""

def plan_step(query: str, observation: str = "") -> str:
    # prompt = f'''
    # You are an AI agent.
    #
    # Decide next action.
    #
    # Actions:
    # - search → if you need more information
    # - answer → if you have enough information
    #
    # Previous observation:
    # {observation}
    #
    # Query:
    # {query}
    #
    # Return ONLY one word: search or answer
    # '''

    # defining a parser since planner now returns a json
    def parse_action(output: str):
        try:
            # extract JSON block if extra text exists
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("action", "search")
        except:
            pass

        # fallback
        if "answer" in output.lower():
            return "answer"
        return "search"

    def is_followup(query: str):
        return any(word in query.lower() for word in [
            "explain", "simplify", "summarize", "in simple terms", "example", "it", "that"
        ])

    prompt = f"""
    Return a JSON with ONE key: action.
    
    Valid values:
    - "search"
    - "answer"
    
    Rules:
    - Use "search" for factual queries
    - Use "answer" for follow-ups or reasoning
    
    Query: {query}
    Observation: {observation}
    
    Output ONLY JSON.
    """

    out = planner(prompt, max_length=10, temperature=0)[0]["generated_text"]
    # print("raw planner output:", out)
    logger.info(f"raw planner output={out}")
    action = parse_action(out)
    # print("parsed action:", action)
    logger.info(f"parsed action={action}")

    # adding a fallback when answer is returned due to memory, but ideally should be search
    # ONLY force search if NO observation yet (first step)
    if action == "answer" and not is_followup(query) and observation == "":
        action = "search"

    return action
    # if "search" in res:
    #     return "search"
    # else:
    #     return "answer"
