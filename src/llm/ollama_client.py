import requests

def check_ollama(host: str) -> bool:
    try:
        r = requests.get(host, timeout=1)
        return r.status_code == 200
    except Exception:
        return False

def chat(model: str, prompt: str, temperature: float = 0.1, num_predict: int = 512) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": num_predict}
    )
    return (resp.get("message", {}) or {}).get("content", str(resp))