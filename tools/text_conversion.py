import re
import pandas as pd
from bert_score import score


def summarization(df, group_cols, text_col, *replace_strings):
    summarized = df.groupby(group_cols, as_index=False).agg({text_col: " ".join})

    for string in replace_strings:
        summarized[text_col] = (
            summarized[text_col].str.lower().replace(string, "", regex=True)
        )

    return summarized


def get_summary(df, client, model="gpt-3.5-turbo-1106", seed=None, scores=False):
    def generate_summary(text):
        response = client.chat.completions.create(
            model=model,
            seed=seed,
            messages=[
                {
                    "role": "system",
                    "content": "Выдели из полученного текста только важные для сервисного центра по ремонту бытовой техники, куда обращается клиент данным текстом, фразы с вопросами, запросами клиента, только когда он хочет что-то выяснить, обращается по поводу какой-то проблемы, заявляет о ней. НЕ ВЫВОДИ НЕНУЖНЫЕ ДЕТАЛИ, такие как адреса, время, телефоны, номера и тому подобное. ПЕРЕЧИСЛЕННОЕ - ЛИШНЯЯ ИНФОРМАЦИЯ. Твоя конечная цель - донести до руководства, с какими запросами от клиентов в первую очередь сталкиваются сотрудники компании. Выводи одной строкой, но состоящей из ОТДЕЛЬНЫХ уникальных предложений, каждое из которых будет содержать весь необходимый контекст, чтобы взглянув на предложение, можно было понять, о чём речь, не видя остальных прердложений. Для этого сначала перефразируй каждую фразу в отдельное предложение так, чтобы оно выглядело понятным и самодостаточным, но используй только исходную смысловую информацию во всём тексте, не придумывай НИКАКУЮ свою. ТОЛЬКО если ты не можешь выделить требуемую информацию, вместо самостоятельно генерируемого ответа выводи: Нет ключевой информации.",
                },
                {
                    "role": "user",
                    "content": "але але да але але да да да сегодня бедняки да подъезжайте нет проблем да да да но вы поняли что у нас каждая дверь морозильной камеры или сама русловая уплотнительная резинка нужда нет ну в этот самый раз и фрагменты и у неё там ну вы же только бутылку хотите посмотреть ничего не делая как я вам могу скинуть размеры я в интернете смотрела да хорошо хорошо проще.",
                },
                {
                    "role": "assistant",
                    "content": "У нас проблема с уплотнительной резинкой или дверью морозильной камеры. Вы только бутылку хотите посмотреть, как я могу вам передать размеры?",
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content

    def compute_scores(row):
        P, R, F1 = score([row["result"]], [row["text"]], lang="ru")
        return pd.Series([P.item(), R.item(), F1.item()])

    df["result"] = df["text"].apply(generate_summary)
    if scores:
        df[["precision", "recall", "f1"]] = df.apply(compute_scores, axis=1)

    return df


def clean_text(df, first_filter, second_filter, third_filter):
    def re_text(text):
        text = text.replace('"', "")
        text = text.replace("\n", " ")
        text = re.sub(r"\b\d\.\s*", "", text)
        text = re.sub(r"(?<=[\.?])\s+", "", text)
        text = re.sub(r"(?<!\s)-\s", "", text)
        fragments = re.split(r"(?<=[\.?])", text)
        fragments = [frag.lstrip() for frag in fragments if frag.strip()]
        return fragments

    df = df[~df.result.str.contains(first_filter, case=False)].reset_index(drop=True)

    fragments = df.result.apply(re_text)
    clean = pd.DataFrame(
        {
            "linkedid": df.linkedid.repeat(fragments.apply(len)).values,
            "text": [frag for list in fragments for frag in list],
        }
    )
    clean = clean[~clean.text.str.contains(second_filter, case=False)]
    clean = clean[clean.text.str.contains(third_filter, regex=True)].reset_index(
        drop=True
    )

    return clean


def clean_json(data):
    pattern = re.compile(r"\[\n{\n.*?model': '.*?'", re.DOTALL)
    cleaned_data = []

    for item in data:
        updated_item = {}
        for key, value in item.items():
            new_value = re.sub(pattern, "", value)
            updated_item[key] = new_value
        cleaned_data.append(updated_item)

    return cleaned_data


def parse_data(text, *keys):
    values = []
    for key in keys:
        match = re.search(rf"'{key}': '([^']+)'", text)
        values.append(match.group(1) if match else None)
    return values
