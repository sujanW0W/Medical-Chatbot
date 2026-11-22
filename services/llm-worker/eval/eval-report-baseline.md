# RAG Pipeline Evaluation Report — baseline

Generated: 2026-11-21 19:58

## Summary

| Metric | Value |
|---|---|
| Total queries | 20 |
| Errors | 0 |
| Router accuracy | 100.0% |
| Avg retrieval relevance (in-KB queries) | 0.671 |
| Avg judge faithfulness (1-5) | 4.05 |
| Avg judge answer relevance (1-5) | 4.8 |
| Avg latency (s) | 28.94 |

## Per-query results

| Query | Expected | Actual | Match | Retrieval Relevance | Faithfulness | Answer Relevance | Latency (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| What are the common symptoms of type 2 diabetes? | retrieve | retrieve | ✅ | 0.65 | 3 | 5 | 21.85 | The answer includes symptoms from the context but adds unmentioned symptoms like 'tingling or numbness' and 'increased hunger' not supported by the provided documents, affecting faithfulness, while still directly addressing the question. |
| What is the function of the pancreas in the human body? | retrieve | retrieve | ✅ | 0.71 | 4 | 5 | 22.41 | The answer accurately addresses the pancreas's functions but includes specific hormone names (insulin, glucagon) not explicitly mentioned in the context, which slightly reduces faithfulness. |
| What causes pneumonia and how is it typically treated? | retrieve | retrieve | ✅ | 0.66 | 4 | 5 | 38.88 | The answer accurately covers causes (bacteria, viruses, fungi, parasites) and treatments from the context but omits 'amoebae' mentioned in the context, and includes some specific treatment examples not explicitly stated. |
| What are the stages of chronic kidney disease? | retrieve | retrieve | ✅ | 0.66 | 5 | 3 | 20.1 | The answer accurately reflects the context's lack of stage-specific information and avoids fabricating details, but only partially addresses the question by not explicitly listing the stages. |
| What is the difference between a virus and a bacterium? | retrieve | retrieve | ✅ | 0.57 | 4 | 5 | 40.25 | The answer accurately compares viruses and bacteria using context-provided information about replication, treatment, and transmission but adds some general biological details not explicitly in the context. |
| What are the common symptoms of hypothyroidism? | retrieve | retrieve | ✅ | 0.73 | 5 | 5 | 27.93 | The answer comprehensively lists all symptoms from the provided context without adding unsupported claims, directly addressing the user's question about hypothyroidism symptoms. |
| How does the human immune system respond to an infection? | retrieve | retrieve | ✅ | 0.65 | 5 | 5 | 24.42 | The answer accurately reflects the context's information about immune response mechanisms and directly addresses the question about how the immune system responds to infections. |
| What are the risk factors for coronary artery disease? | retrieve | retrieve | ✅ | 0.75 | 3 | 5 | 25.85 | The answer addresses the question but introduces modifiable risk factors (e.g., smoking, diet) not explicitly mentioned in the provided context, reducing faithfulness. |
| What is the function of insulin in blood sugar regulation? | retrieve | retrieve | ✅ | 0.66 | 5 | 5 | 29.04 | The answer is fully grounded in the provided context and directly addresses the question about insulin's role in blood sugar regulation without introducing unsupported claims. |
| What are the typical symptoms and treatment of appendicitis? | retrieve | retrieve | ✅ | 0.70 | 4 | 5 | 31.81 | The answer accurately reflects the context's information on symptoms, treatment, and complications but adds specific details (e.g., temperature thresholds, surgical methods) not explicitly mentioned in the provided documents. |
| What causes migraines and what are common treatments? | retrieve | retrieve | ✅ | 0.67 | 4 | 5 | 23.47 | The answer accurately reflects the context's information on migraine causes and treatments, but includes specific drug examples not explicitly mentioned in the provided documents, slightly reducing faithfulness. |
| What is the difference between type 1 and type 2 diabetes? | retrieve | retrieve | ✅ | 0.64 | 3 | 5 | 33.77 | The answer addresses the question about differences between Type 1 and Type 2 diabetes but includes details (e.g., autoimmune destruction, specific symptoms) not explicitly supported by the provided context, reducing faithfulness. |
| What is the latest FDA-approved drug for Alzheimer's disease this year? | web_search | web_search | ✅ | — | 3 | 4 | 22.4 | The answer references Lecanemab (Leqembi) and its 2023 approval, which is not explicitly stated in the provided context, relying partially on external knowledge. The answer addresses the question but introduces details not directly supported by the context. |
| What are the current CDC guidelines for this year's flu vaccine? | web_search | web_search | ✅ | — | 5 | 5 | 29.95 | The answer accurately reflects the CDC guidelines from the provided context, including vaccination recommendations, safety, and timing, without adding unsupported claims. |
| What is the most recent WHO guidance on mpox outbreaks? | web_search | web_search | ✅ | — | 5 | 5 | 33.06 | The answer accurately reflects the context's mention of WHO's focus on Clade 1b, mortality rates, and global action, while appropriately noting the absence of direct guidance documents in the provided sources. |
| What are the newest weight-loss medications approved this year? | web_search | web_search | ✅ | — | 3 | 4 | 19.57 | The answer includes some information not explicitly stated in the context (e.g., 'fintiba') and mentions medications like semaglutide and liraglutide without specifying their 2024 approval status, which slightly reduces faithfulness, while still addressing the user's question about recent weight-loss medications. |
| What is the current recommended COVID-19 booster schedule? | web_search | web_search | ✅ | — | 5 | 5 | 29.29 | The answer accurately reflects the context's details about 2026-2027 vaccine recommendations, booster schedules, timing, and vaccine types without adding unsupported claims, and directly addresses the user's question about the booster schedule. |
| Are there any recent recalls of blood pressure medications? | web_search | web_search | ✅ | — | 5 | 5 | 35.73 | The answer accurately reflects the context's mention of the 2026 Amlodipine recall and past recalls like Valsartan, while clarifying temporal discrepancies and providing relevant guidance. |
| What is the latest research on GLP-1 drugs for diabetes treatment? | web_search | web_search | ✅ | — | 3 | 5 | 40.59 | The answer incorporates details not present in the context, such as the dual-action GLP-1/GIP agonist and cardiovascular risk reduction, which are fabricated claims. However, it effectively addresses the user's question about recent GLP-1 research. |
| What are this year's updated dietary guidelines from the USDA? | web_search | web_search | ✅ | — | 3 | 5 | 28.49 | The answer addresses the question about the 2025–2030 USDA dietary guidelines but includes unsupported details (e.g., sustainable food systems, personalized nutrition) not explicitly stated in the context, while accurately reflecting core guidelines like whole foods and the New Pyramid. |
