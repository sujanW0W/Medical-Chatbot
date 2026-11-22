# RAG Pipeline Evaluation Report

Generated: 2026-11-21 17:35

## Summary

| Metric | Value |
|---|---|
| Total queries | 20 |
| Errors | 0 |
| Router accuracy | 100.0% |
| Avg retrieval relevance (in-KB queries) | 0.671 |
| Avg judge faithfulness (1-5) | 3.45 |
| Avg judge answer relevance (1-5) | 4.75 |
| Avg latency (s) | 31.52 |

## Per-query results

| Query | Expected | Actual | Match | Retrieval Relevance | Faithfulness | Answer Relevance | Latency (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| What are the common symptoms of type 2 diabetes? | retrieve | retrieve | ✅ | 0.65 | 3 | 5 | 19.89 | The answer includes symptoms not mentioned in the context, such as tingling or numbness, but directly addresses the question about Type 2 diabetes symptoms. |
| What is the function of the pancreas in the human body? | retrieve | retrieve | ✅ | 0.71 | 3 | 5 | 50.45 | The answer accurately addresses the pancreas's functions based on the context but includes inferred endocrine details (insulin/glucagon) not explicitly stated in the provided docum… |
| What causes pneumonia and how is it typically treated? | retrieve | retrieve | ✅ | 0.66 | 5 | 5 | 37.0 | The answer accurately reflects the context's information on pneumonia causes (bacteria, viruses, fungi, etc.) and treatment options (antibiotics, antivirals, supportive care), whil… |
| What are the stages of chronic kidney disease? | retrieve | retrieve | ✅ | 0.66 | 3 | 4 | 18.23 | The answer acknowledges the context's lack of explicit stage details but introduces external staging information not present in the provided documents, partially compromising faith… |
| What is the difference between a virus and a bacterium? | retrieve | retrieve | ✅ | 0.57 | 4 | 5 | 28.29 | The answer accurately reflects the context's core differences (independence, reproduction, treatment) but adds some unmentioned details (structure, specific examples), though it re… |
| What are the common symptoms of hypothyroidism? | retrieve | retrieve | ✅ | 0.73 | 5 | 5 | 30.35 | The answer comprehensively lists all symptoms from the provided context without adding unverified information and directly addresses the question about hypothyroidism symptoms. |
| How does the human immune system respond to an infection? | retrieve | retrieve | ✅ | 0.65 | 3 | 5 | 33.02 | The answer includes unsupported claims like memory cells and T cells not explicitly mentioned in the context, but directly addresses the question about immune system responses. |
| What are the risk factors for coronary artery disease? | retrieve | retrieve | ✅ | 0.75 | 4 | 5 | 38.0 | The answer accurately incorporates non-modifiable risk factors from the context but introduces modifiable factors (e.g., smoking, physical inactivity) not explicitly mentioned in t… |
| What is the function of insulin in blood sugar regulation? | retrieve | retrieve | ✅ | 0.66 | 5 | 5 | 30.29 | The answer accurately reflects the context's details about insulin's role in glucose uptake, insulin resistance, hyperinsulinemia, and metabolic effects, while directly addressing … |
| What are the typical symptoms and treatment of appendicitis? | retrieve | retrieve | ✅ | 0.70 | 4 | 5 | 27.54 | The answer is mostly faithful to the context, providing accurate symptoms and treatments mentioned in the documents, though it includes some specific details (e.g., temperature ran… |
| What causes migraines and what are common treatments? | retrieve | retrieve | ✅ | 0.67 | 2 | 5 | 31.32 | The answer includes fabricated details like specific NSAIDs/triptans and CAM therapies not mentioned in the context, but directly addresses the question about causes and treatments… |
| What is the difference between type 1 and type 2 diabetes? | retrieve | retrieve | ✅ | 0.64 | 2 | 5 | 23.94 | The answer includes accurate distinctions between type 1 and type 2 diabetes but fabricates details like the autoimmune cause of type 1 diabetes, which are not explicitly stated in… |
| What is the latest FDA-approved drug for Alzheimer's disease this year? | web_search | web_search | ✅ | — | 3 | 5 | 23.53 | The answer includes specific drug names (e.g., Aducanumab, Donepezil) not explicitly mentioned in the provided context, but it directly addresses the user's question about FDA-appr… |
| What are the current CDC guidelines for this year's flu vaccine? | web_search | web_search | ✅ | — | 5 | 5 | 26.91 | The answer accurately reflects the CDC guidelines from the provided context, including vaccine recommendations, effectiveness, timing, and special considerations, without introduci… |
| What is the most recent WHO guidance on mpox outbreaks? | web_search | web_search | ✅ | — | 5 | 5 | 28.79 | The answer accurately reflects the context's latest information on WHO mpox guidance, including the PHEIC lifting, clade Ib transmission, and cultural reporting considerations, wit… |
| What are the newest weight-loss medications approved this year? | web_search | web_search | ✅ | — | 3 | 2 | 35.65 | The answer includes medications approved in 2021 and 2023, which are not the 'newest this year' (2024), and omits any mention of 2024 approvals, making it partially faithful but no… |
| What is the current recommended COVID-19 booster schedule? | web_search | web_search | ✅ | — | 1 | 5 | 23.8 | The answer provides detailed booster schedules but is not grounded in the provided context, which lacks specific information about COVID-19 booster recommendations. |
| Are there any recent recalls of blood pressure medications? | web_search | web_search | ✅ | — | 3 | 4 | 29.19 | The answer includes information not present in the context (e.g., Valsartan recall) and questions the validity of the MAR-Amlodipine recall's future date, but it still addresses th… |
| What is the latest research on GLP-1 drugs for diabetes treatment? | web_search | web_search | ✅ | — | 1 | 5 | 65.36 | The answer is not grounded in the provided context (which contained an error), but it directly addresses the question about GLP-1 drugs for diabetes treatment. |
| What are this year's updated dietary guidelines from the USDA? | web_search | web_search | ✅ | — | 5 | 5 | 28.88 | The answer accurately reflects the context's details about the 2025–2030 Dietary Guidelines, their focus on whole foods, and resources like the New Pyramid, without adding unsuppor… |
