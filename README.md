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