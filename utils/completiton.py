def generate_answer(prompt, client, model="models/gemini-flash-latest"):
    response = client.models.generate_content(model=model, contents=prompt)
    return getattr(response, "text", str(response))
