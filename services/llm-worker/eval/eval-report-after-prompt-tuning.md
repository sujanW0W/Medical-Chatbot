# RAG Pipeline Evaluation Report — after-prompt-tuning

Generated: 2026-11-21 22:43

## Summary

| Metric | Value |
|---|---|
| Total queries | 20 |
| Errors | 0 |
| Router accuracy | 100.0% |
| Avg retrieval relevance (in-KB queries) | 0.671 |
| Avg judge faithfulness (1-5) | 4.7 |
| Avg judge answer relevance (1-5) | 4.65 |
| Avg latency (s) | 25.8 |

## Per-query results

| Query | Expected | Actual | Match | Retrieval Relevance | Faithfulness | Answer Relevance | Latency (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| What are the common symptoms of type 2 diabetes? | retrieve | retrieve | ✅ | 0.65 | 5 | 5 | 18.24 | The answer accurately lists symptoms from the context and explains detection and symptom variation without adding unsupported claims, fully addressing the question. |
| What is the function of the pancreas in the human body? | retrieve | retrieve | ✅ | 0.71 | 3 | 5 | 25.98 | The answer correctly identifies the pancreas's exocrine and endocrine functions based on the context but adds specific hormones (insulin/glucagon) not explicitly mentioned in the provided documents, slightly reducing faithfulness. |
| What causes pneumonia and how is it typically treated? | retrieve | retrieve | ✅ | 0.66 | 5 | 5 | 18.38 | The answer accurately reflects the context's information about causes and acknowledges treatment limitations, while addressing both parts of the question. |
| What are the stages of chronic kidney disease? | retrieve | retrieve | ✅ | 0.66 | 5 | 2 | 13.13 | The answer accurately reflects the context's lack of information on CKD stages but fails to address the user's question about the stages themselves. |
| What is the difference between a virus and a bacterium? | retrieve | retrieve | ✅ | 0.57 | 5 | 5 | 13.2 | The answer accurately reflects the context provided, directly addressing the question with key differences between viruses and bacteria without introducing unsupported claims. |
| What are the common symptoms of hypothyroidism? | retrieve | retrieve | ✅ | 0.73 | 5 | 5 | 22.7 | The answer accurately lists symptoms and complications from the provided context without adding unsupported claims, and directly addresses the user's question about hypothyroidism symptoms. |
| How does the human immune system respond to an infection? | retrieve | retrieve | ✅ | 0.65 | 5 | 5 | 17.02 | The answer accurately reflects the context provided, covering inflammation, antibodies, phagocytes, complement system, and adaptive immune responses without introducing unsupported claims. |
| What are the risk factors for coronary artery disease? | retrieve | retrieve | ✅ | 0.75 | 5 | 5 | 19.89 | The answer accurately reflects the provided context without adding external information and directly addresses the question about coronary artery disease risk factors. |
| What is the function of insulin in blood sugar regulation? | retrieve | retrieve | ✅ | 0.66 | 5 | 5 | 17.71 | The answer accurately reflects the context's information about insulin's role in glucose regulation, including its secretion, receptor interaction, and consequences of insulin deficiency or resistance. |
| What are the typical symptoms and treatment of appendicitis? | retrieve | retrieve | ✅ | 0.70 | 5 | 3 | 39.32 | The answer accurately reflects the context's content about treatment and rupture symptoms but omits typical appendicitis symptoms like abdominal pain, which are not explicitly mentioned in the provided documents. |
| What causes migraines and what are common treatments? | retrieve | retrieve | ✅ | 0.67 | 5 | 5 | 24.95 | The answer accurately reflects the causes and treatments from the provided context without introducing unsupported information or偏离 the question. |
| What is the difference between type 1 and type 2 diabetes? | retrieve | retrieve | ✅ | 0.64 | 5 | 5 | 18.0 | The answer accurately reflects the context's details about causes, treatments, and characteristics of type 1 and type 2 diabetes without introducing unsupported information. |
| What is the latest FDA-approved drug for Alzheimer's disease this year? | web_search | web_search | ✅ | — | 5 | 5 | 21.62 | The answer accurately reflects the context's lack of specific drug name while addressing the question about FDA-approved Alzheimer's drugs. |
| What are the current CDC guidelines for this year's flu vaccine? | web_search | web_search | ✅ | — | 5 | 5 | 22.91 | The answer accurately reflects the context provided, addressing the 2025–2026 flu season guidelines and clarifying the lack of 2023-specific data, while directly responding to the user's query about current CDC recommendations. |
| What is the most recent WHO guidance on mpox outbreaks? | web_search | web_search | ✅ | — | 5 | 5 | 28.92 | The answer accurately references the provided context, citing the interim guidance and PHEIC lifting without adding unverified claims, and directly addresses the user's query about recent WHO mpox outbreak guidance. |
| What are the newest weight-loss medications approved this year? | web_search | web_search | ✅ | — | 2 | 3 | 53.46 | The answer fabricates the drug name 'Retatrutide' not mentioned in the context and misrepresents the timeline, while partially addressing the question about new weight-loss medications. |
| What is the current recommended COVID-19 booster schedule? | web_search | web_search | ✅ | — | 5 | 5 | 19.09 | The answer correctly acknowledges the lack of current booster schedule information in the provided context and directs to official sources, aligning with the context's limitations and addressing the user's question. |
| Are there any recent recalls of blood pressure medications? | web_search | web_search | ✅ | — | 5 | 5 | 45.72 | The answer accurately reflects the context provided, citing specific medications and recall details without adding unverified information, and directly addresses the user's question about recent blood pressure medication recalls. |
| What is the latest research on GLP-1 drugs for diabetes treatment? | web_search | web_search | ✅ | — | 4 | 5 | 39.31 | The answer accurately reflects the context about tirzepatide, BMJ study, and expanding trials but introduces a minor fabricated claim about 'newer formulations' not explicitly mentioned in the sources. |
| What are this year's updated dietary guidelines from the USDA? | web_search | web_search | ✅ | — | 5 | 5 | 36.39 | The answer accurately reflects the context's details about the 2025–2030 Dietary Guidelines, their focus on whole foods, and provides correct resources without adding unverified information. |
