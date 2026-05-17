from transformers import pipeline

decider = pipeline("text2text-generation", model="google/flan-t5-small")

def decide_tool(query: str):
    prompt = f"""
    Classify the query into ONE of the following labels:
    
    math
    search
    
    Rules:
    - math → arithmetic expressions like 2+2, 10/5, (2+3)*4
    - search → general knowledge, facts, or questions
    
    Examples:
    
    Query: 2+2
    Answer: math
    
    Query: 10/5 + 3
    Answer: math
    
    Query: (2+3)*4
    Answer: math
    
    Query: what is machine learning
    Answer: search
    
    Query: who is Elon Musk
    Answer: search
    
    Query: latest AI trends
    Answer: search
    
    Now classify:
    
    Query: {query}
    Answer:
    """

    results = decider(prompt, max_length=50)
    res_parsed = results[0]['generated_text']
    print(res_parsed)

    # mapping synonyms sincer small LLM not ahering to strict return rules
    res = res_parsed.strip().lower()

    if any(word in res for word in ["math", "arithmetic", "calculation"]):
        return "math"
    elif any(word in res for word in ["search", "knowledge", "information"]):
        return "search"
    else:
        return "unspecified"
