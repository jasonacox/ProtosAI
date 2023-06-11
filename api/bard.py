import requests

def get_bard_response(question):
    url = "https://bard.ai/api/v1/generate"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
    }
    data = {
        "prompt": question,
        "temperature": 0.9,
        "max_tokens": 100,
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()["text"]

def main():
    while True:
        question = input("What is your question (type 'exit' to end)? ")
        if question == "exit":
            break
        response = get_bard_response(question)
        print(f"Bard: {response}")

if __name__ == "__main__":
    main()

