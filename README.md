## Clustering and defining text categories 
The presented examples demonstrate how LLM can be utilized for:
* Extracting the brief essence from texts
* Clustering texts into categories based on their content
* Forming descriptions and characteristics of categories

---

### Objective

The results obtained can be leveraged by businesses, for instance, to understand the most common inquiries made to customer service centers or technical support by clients and company employees.

---

### Used tools

[GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo) and [GPT 4](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) were used depending on the volume of texts and the complexity of the task, as well as the final processing cost.

Additionally, on large datasets, [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) was employed for clustering and [RuBERT tiny 2](https://huggingface.co/cointegrated/rubert-tiny2) was used for generating text embeddings.

---

## Receiving Q&A file based on Telegram messages
### OpenAI API key setup
To get image descriptions from your chat, first, you need to set your OpenAI [API key](https://platform.openai.com/api-keys) environment variable on your OS.
Just run the following [script](https://github.com/Darveivoldavara/clustering_and_naming_categories/blob/main/setup_openai_key.sh) in your command line and specify your API key:

```
bash setup_openai_key.sh
```
### Telegram message history export
To retrieve your chat history in Telegram, go to the chat interface, click on the three dots for options at the top right corner, and select "Export chat history".
Next, make sure to select **"Format": JSON** and other necessary parameters as needed. Specify the save path as **"Path"** to the root of this project, and you will have a similar folder named [*source*](https://github.com/Darveivoldavara/clustering_and_naming_categories/tree/main/source) with chat data.

![Без имени](https://github.com/Darveivoldavara/clustering_and_naming_categories/assets/101942420/8693c989-6065-4c89-b99e-635e4d9d656e)

### Retrieving Q&A file
Then, you can run qa_extract.py:

```
python3 qa_extract.py
```

and the resulting **qa.json** file will appear in the [*data*](https://github.com/Darveivoldavara/clustering_and_naming_categories/tree/main/data) folder.
