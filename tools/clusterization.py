import os
import json

import numpy as np
import pandas as pd
import plotly.express as px

from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def get_clusters(matrix, n_clusters=2, seed=None):
    clusterer = KMeans(
        n_clusters=n_clusters,
        max_iter=100,
        init="random",
        n_init=10,
        random_state=seed,
        algorithm="lloyd",
    )
    clusterer.fit(matrix)

    return clusterer.labels_


def generate_topics(
    client,
    df,
    col,
    matrix,
    n_clusters=2,
    rev_per_cluster=30,
    model="gpt-3.5-turbo-1106",
    seed=None,
):
    messages = [
        {
            "role": "system",
            "content": "Ты - профессиональный маркетолог с многолетним стажем. Ты специализируешься на выявлении и характеризации ключевых особенностей взаимодействия пользователей, клиентов с продуктами компаний, бизнесом. Я готов заплатить тебе за хорошее правильное решение до 200$ в зависимости от его качества. Далее представлены фрагменты диалогов клиентов с сервисным центром по ремонту бытовой техники. Эти фрагменты уже разделены на несколько указанных кластеров. Сформулируй описание, название для каждого кластера так, чтобы легко было понятно, что его выделяет, характеризует среди остальных кластеров. Ответ дай в виде подобной JSON структуры, только с двойными кавычками: {'Кластер 0': 'Название 0', 'Кластер 1': 'Название 1'} и так далее.",
        }
    ]
    message = {"role": "user", "content": ""}

    tsne = TSNE(random_state=seed)
    vis_dims2 = tsne.fit_transform(matrix)

    for i in range(n_clusters):
        cluster_df = df[df[col] == i].reset_index(drop=True)

        cluster_center = vis_dims2[cluster_df.index].mean(axis=0)

        distances = np.sqrt(
            ((vis_dims2[cluster_df.index] - cluster_center) ** 2).sum(axis=1)
        )
        closest_indices = distances.argsort()[:rev_per_cluster]

        closest_reviews = cluster_df.iloc[closest_indices].text

        reviews = "\n ".join(closest_reviews.values)
        message["content"] += f"\n Кластер {i}: {reviews} "

    messages.append(message)

    response = client.chat.completions.create(
        model=model,
        seed=seed,
        messages=messages,
        response_format={"type": "json_object"},
    )
    topics_json = response.choices[0].message.content

    topics_dict = json.loads(topics_json)
    return topics_dict


def llm_clusterization(qa, model="gpt-3.5-turbo-1106", seed=None):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    messages = [
        {
            "role": "system",
            "content": "Ты - профессиональный маркетолог с многолетним стажем. Ты специализируешься на выявлении и характеризации ключевых особенностей взаимодействия сотрудников с продуктами, приложениями компании. Я готов заплатить тебе за хорошее правильное решение до 200$ в зависимости от его качества. Далее представлены пары вопрос-ответ с их id из обращений мастеров сервисного центра по ремонту бытовой техники в техническую поддержку. УНИКАЛЬНО, чтобы каждая пара принадлежала только одному кластеру, раздели ВСЕ пары на конкретезированные, но не повторяющиеся кластеры по их смысловому содержанию, конкретной проблеме, с которой обращается мастер. Не формируй обобщённые кластеры, не объединённые общей проблемой. Также сформулируй описание, название для каждого кластера так, чтобы легко было понятно, что его выделяет, характеризует среди остальных кластеров. Ответ дай в виде ТОЛЬКО подобной JSON структуры: {'Название кластера': ['id пары', 'id пары'], 'Название кластера: ['id пары']} и так далее. В названии кластера не должно быть самого слова 'Кластер', id пары указывай просто одним числом.",
        },
        {"role": "user", "content": f"{qa}"},
    ]

    response = client.chat.completions.create(
        model=model,
        seed=seed,
        messages=messages,
        response_format={"type": "json_object"},
    )
    clusters_json = response.choices[0].message.content

    clusters_dict = json.loads(clusters_json)
    return clusters_dict


def plot_clusters(title, df, col, plot_text_size, legend_text_size, clusters=None):
    if clusters:
        names = df[col].replace(clusters)
    else:
        names = df[col]
    fig = (
        px.pie(df, names=names)
        .update_traces(
            textinfo="percent",
        )
        .update_layout(
            uniformtext_minsize=plot_text_size,
            uniformtext_mode="hide",
            width=2100,
            height=900,
            title=dict(text=title, x=0.5, y=0.98, font_size=50),
            legend=dict(font_size=legend_text_size, y=0.5, yanchor="middle"),
        )
    )
    return fig
