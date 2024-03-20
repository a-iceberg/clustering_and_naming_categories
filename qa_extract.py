import json
import requests
import os
import re
import base64
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def photo_description(message):
    user_text = ""
    file_path = "../source/" + message["photo"]
    model = "gpt-4-vision-preview"

    base64_image = encode_image(file_path)
    logger.info(f"base64_image file_path: {file_path} len: {len(base64_image)}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Максимально детально опиши, что ты видишь на скриншоте экрана. При наличии конкретных показателей, как, например, заряд батареи, уровень сигнала и тому подобных, обязательно точно указывай их, если возможно, в числовых значениях. Если присутствует переписка, предоставляй её содержание полностью дословно.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1500,
    }

    logger.info("Posting payload..")
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
    except:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )

    if response.status_code == 200:
        response_json = response.json()
        description = response_json["choices"][0]["message"]["content"]
        user_text += "\nОписание присланного скриншота: "
        user_text += description
        logger.info(f"Screenshot description:\n{description}")

    else:
        logger.error(f"Error fetching image description: {response.text}")
    return user_text


def extract_text(message):
    message_text = ""

    if "photo" in message:
        message_text += photo_description(message)
        if message["text"] != "":
            message_text += f'\nКомментарий к скриншоту: {message["text"]}'

    elif isinstance(message["text"], str) and message["from"] != "mrmbot":
        message_text += message["text"]

    elif isinstance(message["text"], list) and (
        message["from"] != "mrmbot"
        or (message["from"] == "mrmbot" and "result" in message["text"][0])
    ):
        full_text = []
        if message["from"] == "mrmbot":
            full_text.append("\nТехническая информация о пользователе: ")

        pattern = r"\n},"

        for text_component in message["text"]:
            if (
                isinstance(text_component, dict)
                and "text" in text_component
                and text_component["type"] != "link"
            ):
                full_text.append(text_component["text"])

            elif isinstance(text_component, str):
                parts = re.split(pattern, text_component, maxsplit=1)
                cleaned_text = parts[0]
                if cleaned_text:
                    full_text.append(cleaned_text)

        message_text += " ".join(full_text)
    return message_text


def main():
    file_path = "../source/result.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    questions = {}
    answers = {}

    for message in data["messages"]:
        if message["type"] == "message":
            message_text = extract_text(message)

            if message_text == "":
                continue

            if "forwarded_from" in message:
                question_id = message["id"]
                questions[question_id] = message_text

            elif "reply_to_message_id" in message:
                reply_id = message["reply_to_message_id"]
                if reply_id in answers:
                    if (
                        not "Текст: " in answers[reply_id]
                        and not "Описание присланного скриншота: " in message_text
                    ):
                        answers[reply_id] += f" \nТекст: {message_text}"
                    else:
                        answers[reply_id] += " " + message_text

                elif (
                    "Комментарий к скриншоту: " in message_text
                    or "Описание присланного скриншота: " in message_text
                    or "Техническая информация о пользователе: " in message_text
                ):
                    answers[reply_id] = message_text

                else:
                    answers[reply_id] = f"\nТекст: {message_text}"

    qa_pairs = []
    for q_id, question_text in questions.items():
        if question_text == "":
            continue

        answer = answers.get(q_id, "No answer found")
        qa_pairs.append({f"Question {q_id}": question_text, f"Answer {q_id}": answer})

    with open("data/qa.json", "w", encoding="utf-8") as file:
        json.dump(qa_pairs, file, ensure_ascii=False, indent=4)

    print("Extracted", len(qa_pairs), "question-answer pairs.")

    return qa_pairs


if __name__ == "__main__":
    main()
