def generate_prompt(context, user_query):

    prompt = f"""You are a legal document analyzer. Based on the following excerpts from a legal document, answer the user's question accurately and concisely.
    
    Document Excerpts:
    {context}

    User Question: {user_query}

    Answer:"""
    return prompt