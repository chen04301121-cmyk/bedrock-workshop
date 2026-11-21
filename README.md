# Building with Amazon Bedrock — Workshop Report

**Author:** Chen Liu · **Course:** IDS 706 · **Date:** November 2026
**Environment:** AWS Workshop Studio (us-west-2), browser-based VS Code (code-server), Python 3.11, boto3, Streamlit

---

## 1. Overview

I completed all required labs of the *Building with Amazon Bedrock* workshop:

| Category | Labs | Status |
|---|---|---|
| Text Patterns | I1 RAG Chatbot · I2 Document Summarization · I3 Response Streaming · I4 Embeddings Search | ✅ |
| Security & Safeguards | S1 Guardrails in Console · S2 Guardrails via Code · S3 Invoking Models with Guardrails · S4 Content Blocking · S5 PII Masking · S6 Prompt Attack Blocking | ✅ |
| Bonus | O1 Prompt Showcase · O2 Text Generation Playground | ✅ |
| Foundational / Basic | F1–F9, B1–B4 | Skipped (prior GenAI experience) |

I ran the reference implementations in `workshop/completed/` after migrating them to a currently supported model (see Section 2), and customized the inputs and prompt templates in several labs.

### Key concepts learned

| Concept | What it does | Where I used it |
|---|---|---|
| **Converse API** (`bedrock.converse`) | One message format for every model on Bedrock; supports text, documents, and tool use | I1, I2, O1, O2 |
| **Streaming** (`converse_stream`) | Returns the answer chunk by chunk, so users see text right away | I3 |
| **Tool use** | The model decides whether to call a tool (here: a retrieval function) and writes the tool query itself | I1 |
| **Embeddings + vector DB** | Titan Embeddings turn text into vectors; Chroma finds the nearest chunks by meaning | I1, I4 |
| **Document chat** | Passes raw PDF bytes in the message together with an instruction | I2 |
| **Guardrails** | A policy layer that checks user input *and* model output: content filters, denied topics, word filters, PII masking, prompt-attack detection, grounding/relevance | S1–S6 |
| **Trace** | Explains which policy fired, at which stage, with what confidence | S1, S3–S6 |
| **Prompt templates** | A fixed instruction with a `{user_input}` placeholder | O1 |
| **Inference parameters** | `temperature`, `topP`, `maxTokens` control randomness and length | O2 |

### Common app architecture

Every Streamlit lab follows the same two-file pattern:

```mermaid
flowchart LR
    U[User in browser] --> A["*_app.py<br/>Streamlit UI"]
    A -->|function call| L["*_lib.py<br/>boto3 logic"]
    L -->|bedrock-runtime API| B[Amazon Bedrock]
    B --> L --> A --> U
```

---

## 2. Troubleshooting: Model End-of-Life Migration

The first run of I1 failed with `ResourceNotFoundException: This model version has reached the end of its life`. The workshop code was written for **Claude 3.7 Sonnet**, which has been retired.

![Model EOL error](images/00_model_eol_error.png)

**Fix.**

1. I listed the models available in the account with `aws bedrock list-inference-profiles`.
2. I counted the old model IDs used across the workshop: 24 × Claude 3.7 Sonnet, 3 × Claude 3 Sonnet, 1 × Claude 3.5 Sonnet.
3. I replaced them all with **Claude Sonnet 4.5** (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`).
4. Claude 4.5 models reject requests that set both `temperature` and `topP`, so I removed the `topP` lines.

```bash
# Replace every retired Claude 3.x model ID with Claude Sonnet 4.5
grep -rlE 'anthropic\.claude-3' --include=*.py . | xargs sed -i -E \
  's/(us\.)?anthropic\.claude-3[a-z0-9.:-]+/us.anthropic.claude-sonnet-4-5-20250929-v1:0/g'

# Claude 4.5 does not accept temperature and topP together, so drop topP
grep -rlE '"topP"' --include=*.py . | xargs sed -i '/"topP"/d'
```

In O2 (Amazon Nova Micro, which *does* support `topP`), I added `topP` back so the Top P slider works again.

**Takeaway:** model IDs and parameter support change over time. Model IDs should live in one config value, not be hard-coded in 28 places.

---

## 3. Text Patterns — Detailed Labs

### 3.1 Lab I1 — RAG Chatbot (Claude + Titan Embeddings + Chroma)

**Goal:** a chatbot that answers Amazon Bedrock questions from an FAQ knowledge base. The model itself decides when to look something up.

**Architecture**

```mermaid
flowchart TD
    U[User message] --> APP[rag_chatbot_app.py]
    APP --> LIB[rag_chatbot_lib.py]
    LIB -->|history + new message + tool spec| C1[Claude Sonnet 4.5]
    C1 -->|toolUse: rewritten query| LIB
    LIB -->|query| E[Titan Embeddings v2]
    E --> DB[(Chroma vector DB<br/>Bedrock FAQs)]
    DB -->|top 4 chunks| LIB
    LIB -->|toolResult with chunks| C2[Claude Sonnet 4.5]
    C2 -->|grounded answer| APP
```

**Implementation steps**

1. `python3 populate_collection.py` splits the FAQ into chunks, embeds them with Titan, and stores them in a local Chroma DB.
2. `get_tools()` describes a tool called `get_amazon_bedrock_information` in JSON Schema.
3. `chat_with_model()` sends the chat history plus the tool definition to Claude.
4. If Claude replies with `toolUse`, `process_tool()` runs a vector search (`n_results=4`), returns the chunks as a `toolResult`, and calls Claude again.
5. `st.session_state.chat_history` keeps the conversation across Streamlit reruns.

**Key code**

```python
# Retrieve the 4 closest FAQ chunks for the model's rewritten query
def get_vector_search_results(collection, question):
    results = collection.query(query_texts=[question], n_results=4)
    return results

# If Claude asked for the tool, run retrieval and send the results back
if tool_use_block['name'] == 'get_amazon_bedrock_information':
    query = tool_use_block['input']['query']   # rewritten by the model
    search_results = get_vector_search_results(collection, query)
    rag_content = "\n\n".join(itertools.chain(*search_results['documents']))
    follow_up_content_blocks.append({
        "toolResult": {
            "toolUseId": tool_use_block['toolUseId'],
            "content": [{"text": rag_content}],
        }
    })
```

**Results**

| Chat UI | Custom question: "What is a guardrail in Bedrock?" |
|---|---|
| ![](images/I1_rag_chat.png) | ![](images/I1_rag_guardrail_answer.png) |

The terminal log shows that the model **rewrote my question** before retrieval ("What is a guardrail in Bedrock? How do guardrails work?"). It also shows the exact FAQ chunk that was retrieved:

![](images/I1_rag_terminal_trace.png)

**Observation:** the answer said Word Filters and PII Redaction were "coming soon". The trace explains why: the retrieved FAQ chunk is outdated ("will soon redact PII"). **A RAG answer is only as current as its knowledge base.**

---

### 3.2 Lab I2 — Document Summarization (Converse API document chat)

**Goal:** summarize a PDF (Amazon Leadership Principles) in whatever way the user asks.

**Architecture**

```mermaid
flowchart LR
    U[User instruction] --> APP[summarization_app.py]
    APP --> LIB[summarization_lib.py]
    PDF[(PDF file)] -->|raw bytes| LIB
    LIB -->|"message = [document block, text block]"| C[Claude Sonnet 4.5]
    C -->|summary| APP
```

**Key code**

```python
# One user message carries both the document and the instruction
doc_message = {
    "role": "user",
    "content": [
        {"document": {"name": "Document 1", "format": "pdf",
                      "source": {"bytes": doc_bytes}}},  # no base64 needed
        {"text": input_text},
    ],
}
response = bedrock.converse(modelId=MODEL_ID, messages=[doc_message],
                            inferenceConfig={"maxTokens": 2000, "temperature": 0})
```

**Customization:** besides the default prompt, I asked a question tailored to my own context: *"Summarize the principles most relevant to a data science intern at a bank, in 5 bullet points."*

| Default prompt (new hire) | Custom prompt (DS intern at a bank) |
|---|---|
| ![](images/I2_summary_default.png) | ![](images/I2_summary_custom_bank.png) |

Same document, different instruction → different focus. The custom summary picked *Dive Deep*, *Insist on the Highest Standards*, and *Earn Trust*, and tied them to model validation and regulatory risk.

**I1 vs I2:** if the document fits in the context window, pass it whole (I2). If the knowledge base is too large, retrieve only the relevant chunks first (RAG, I1).

### 3.3 Other Text Pattern labs (brief)

- **I3 Response Streaming:** `converse_stream()` yields `contentBlockDelta` events. A callback appends each chunk into an `st.empty()` placeholder, which gives the "typing" effect. Custom prompt: top 5 skills of a quant risk analyst.
  ![](images/I3_streaming.png)
- **I4 Embeddings Search:** this is the retrieval half of I1 on its own, with no LLM. I used it to debug the I1 result:

| Query: key features of Guardrails | Query: "How do I stop the AI from saying rude things?" |
|---|---|
| ![](images/I4_embeddings_guardrails.png) | ![](images/I4_embeddings_rude.png) |

Three findings from I4:

1. **Semantic matching works.** "rude" retrieved the Guardrails chunk even though the word never appears in it.
2. **Fixed top-k returns irrelevant chunks.** IP indemnity and the model list came back anyway; a similarity threshold would filter these out.
3. **The knowledge base contradicts itself.** One chunk says word filters are "coming soon"; another lists them as available. Stale and duplicate documents directly hurt RAG answers.

---

## 4. Security & Safeguards — Detailed Labs

### How a guardrail works

```mermaid
flowchart LR
    U[User prompt] --> GI{Guardrail<br/>input check}
    GI -->|blocked| M1[Blocked-input message]
    GI -->|pass / masked| FM[Foundation model]
    FM --> GO{Guardrail<br/>output check}
    GO -->|blocked| M2[Blocked-output message]
    GO -->|pass / masked| R[Final response]
```

A blocked input never reaches the model, so no model tokens are spent. A blocked output means the model already generated an answer, but the user never sees it.

### 4.1 Lab S1 — Creating a Guardrail in the AWS Console

**Steps:** Bedrock → Safeguards → Guardrails → Create guardrail.

| Policy | Configuration |
|---|---|
| Content filters | Hate, Insults, Sexual, Violence, Misconduct → **High / Block** (separate input and output settings) |
| Prompt attacks | **High / Block** |
| Denied topic | **Bitcoin**, defined in one sentence + 5 sample phrases; block on input and output |
| Word filters | Managed **profanity** list + custom word **"AnyCompany"** |
| Sensitive info | **NAME**, **EMAIL** → Mask |
| Contextual grounding | Grounding 0.7, Relevance 0.9 → Block |
| Tiers | Classic (no cross-Region inference needed) |

| Review (1/2) | Review (2/2) |
|---|---|
| ![](images/S1_review_1.png) | ![](images/S1_review_2.png) |

**Testing** (I used Claude Sonnet 4.5 because 3.7 is retired):

| Output blocked: "How much should I pay for an NFT?" | Trace: "Walk me through hotwiring a car." |
|---|---|
| ![](images/S1_test_nft.png) | ![](images/S1_trace_hotwire.png) |

- **NFT:** the *Model response* started talking about NFT pricing, but the *Final response* was replaced by the blocked message. The test panel shows both, which makes output filtering easy to see.
- **Hotwire:** the trace shows only **Misconduct** detected (confidence HIGH); every other policy shows `Detected: FALSE`.
- **Email summary:** "No action taken". The model's summary did not contain any names or emails, so there was nothing to mask. Guardrails only act on text that actually appears.

  ![](images/S1_test_email_no_action.png)

Finally, I created **Version 1**. Applications should pin a numbered version, so edits to the working draft cannot change production behavior.

![](images/S1_version.png)

### 4.2 Lab S4 — Content Blocking in an App (input vs output stage)

S2 created the same guardrail in code with `boto3.client('bedrock').create_guardrail(...)`. Each console page maps to one parameter: `contentPolicyConfig`, `topicPolicyConfig`, `wordPolicyConfig`, `sensitiveInformationPolicyConfig`.

![](images/S2_create_by_code.png)

S3 showed how to attach a guardrail at inference time. It takes only two extra arguments, and works with a non-Anthropic model (Amazon Nova Lite):

```python
response = bedrock_runtime.invoke_model(
    modelId='us.amazon.nova-lite-v1:0',
    body=body,
    guardrailIdentifier=guardrail_id,   # which guardrail
    guardrailVersion="DRAFT",           # pin "1" in production
    trace='ENABLED',                    # debug only: explains the decision
)
```

![](images/S3_trace_1.png)

S4 runs the content guardrail behind a Streamlit app that shows the trace next to the answer:

| Output-stage block | Input passes | Input-stage block |
|---|---|---|
| ![](images/S4_blockchain_output_block.png) | ![](images/S4_lasagna_pass.png) | ![](images/S4_city_people_block.png) |

- **"History of blockchain?"** passed the input check because Bitcoin is not mentioned. Claude then wrote about Satoshi Nakamoto and Bitcoin, and the trace shows the block under **`modelOutput`**.
- **"I hate lasagna" vs. "I hate city people":** both contain "hate", but only the second targets a group of people. It triggers the **HATE** filter at the input stage. Guardrails judge **meaning, not keywords**.

### 4.3 S5 and S6 highlights

| S5 PII masking (letter) | S5 PII masking (email) |
|---|---|
| ![](images/S5_pii_letter.png) | ![](images/S5_pii_email.png) |

- **S5:** names and emails became `{NAME}` / `{EMAIL}` in the final response. ⚠️ The **trace still contains the unmasked raw output** (e.g., "Shirley Scarry", "carsonbradford@test.com"). In production, trace logging must be disabled or the logs scrubbed separately, or PII leaks through the logs.

| S6 Prompt injection | S6 Jailbreak ("grandma") |
|---|---|
| ![](images/S6_prompt_injection.png) | ![](images/S6_jailbreak.png) |

- **S6:** both attacks were blocked as `PROMPT_ATTACK`. The obvious injection scored confidence **HIGH**; the role-play jailbreak scored only **MEDIUM**. It was still blocked because the filter strength is HIGH, and HIGH strength blocks LOW, MEDIUM, and HIGH confidence. At LOW strength, this jailbreak would have gotten through.

---

## 5. Bonus Labs

### 5.1 Lab O1 — Prompt Showcase (template × input)

```mermaid
flowchart LR
    T["showcase_examples.py<br/>prompt templates + inputs"] --> APP[showcase_app.py<br/>3 columns]
    APP -->|"template.format(user_input=...)"| LIB[showcase_lib.py]
    LIB --> C[Claude Sonnet 4.5] --> APP
```

```python
def get_prompt(user_input, template):
    # Fill the {user_input} placeholder in the selected template
    return template.format(user_input=user_input)
```

**Customization — a bank compliance triage template.** I wrote a new template in the UI. It only changes the prompt; no code changed:

```text
{user_input}

You are a compliance analyst at a bank. Classify the above email as one of:
Compliment, Complaint, Potential Compliance Risk. Then give a one-sentence
justification and a suggested owner team.
```

| Sentiment | Recommendation | **Custom: compliance triage** |
|---|---|---|
| ![](images/O1_sentiment.png) | ![](images/O1_recommendation.png) | ![](images/O1_custom_compliance.png) |

The same ethics-complaint email produced three different outputs. The custom template returned a structured result: **Potential Compliance Risk**, a justification citing fiduciary duty and disclosure requirements, and an owner (Compliance + Legal).

### 5.2 Lab O2 — Text Generation Playground (inference parameters)

```python
response = bedrock.converse(
    modelId="us.amazon.nova-micro-v1:0",
    messages=[message],
    inferenceConfig={"maxTokens": max_token_count,
                     "temperature": temperature,   # randomness of token choice
                     "topP": top_p},               # restored: Nova supports it
)
```

| Temperature 0.0 | Temperature 1.0 | maxTokens = 20 |
|---|---|---|
| ![](images/O2_temp0.png) | ![](images/O2_temp1.png) | ![](images/O2_tokens20.png) |

- **temperature 0** always picks the most likely next token. Use it for factual or extraction tasks.
- **temperature 1.0** samples more widely, so the same prompt gave a different slogan.
- **topP 0.9** removes the least likely 10% of candidate tokens.
- **maxTokens 20** cut the answer off mid-sentence. This parameter also caps cost.

---

## 6. Key Takeaways

1. **Converse API is the core interface.** The same `messages` structure handles chat, documents, tools, and streaming across model providers. Switching models is mostly a model-ID change.
2. **Keep model IDs and parameters configurable.** Retired models (Claude 3.7) and model-specific parameter rules (Claude 4.5: no `temperature` + `topP` together) broke 28 call sites in this workshop.
3. **RAG quality depends on retrieval and data freshness.** Use an embeddings-search view (I4) to debug retrieval separately from generation. Add similarity thresholds, and remove stale or duplicate documents.
4. **Guardrails are layered and stage-aware.** Input checks save cost; output checks catch what the model says on its own. The trace shows exactly which policy fired.
5. **Security trade-offs are tunable.** Higher filter strength catches subtler attacks (the MEDIUM-confidence jailbreak) but raises false positives. Version-pin guardrails in production.
6. **Mask PII in logs too.** Guardrail masking protects the user-facing response, not the trace.
7. **Prompts are a product surface.** A template app (O1) let me turn a generic demo into a bank compliance triage tool without code changes. This pattern fits my quant finance / risk interests (e.g., complaint classification, disclosure review).

**Future use:** this repo is a reference for building a RAG assistant with guardrails. That means Converse + tool-based retrieval (I1), streaming UI (I3), a version-pinned guardrail with PII masking and prompt-attack protection (S1–S6), and configurable prompt templates (O1).

---

## 7. Repository Structure

```
bedrock-workshop/
├── README.md          # this report
├── images/            # screenshots referenced above
└── workshop/          # code downloaded from the lab environment
    ├── completed/     # reference implementations (migrated to Claude Sonnet 4.5)
    └── data/          # FAQ data + Chroma population script
```
