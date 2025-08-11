# Tabel of contents
- LLM
- RAG
- AI AGENT
- AGENTIC AI


# LLM
## Definition
- Trained on vast text data to learn language patterns.
- Predicts and generates text based on context.
- Powers tasks like Q&A, summarization, translation.  
👉 An AI model trained on massive text data to understand and generate human-like language

<a href="images/ChatgptPipeline.png">
  <img src="images/ChatgptPipeline.png" width="40%">
</a>

Some parts of the diagram may be inaccurate.([Bytebytego](https://blog.bytebytego.com/p/ep-44-how-does-chatgpt-work?utm_source=publication-search))

⚠️ LLMs can’t access real-time info, remember past chats, or reason deeply.  
✅ Solved with RAG, memory modules, function calling, and agents.

## Type of LLMs
| Model Type          | Data Examples                          | How It’s Trained (Simple)                                     | Example Model & Link |
|---------------------|-----------------------------------------|----------------------------------------------------------------|----------------------|
| **Base / Foundation** | Web text, books, code (e.g. CommonCrawl, GitHub, Wikipedia) | Self-supervised learning: next-token prediction on massive unlabeled data ([Medium](https://medium.com/%40yashwanths_29644/llm-finetuning-series-05-llm-architectures-base-instruct-and-chat-models-a6219c39c362), [Wikipedia](https://en.wikipedia.org/wiki/Llama_%28language_model%29)) | GPT‑3 before fine-tuning (OpenAI) |
| **Instruct‑tuned**   | Instruction-output pairs labeled by humans | Fine-tuning on prompt-response dataset; often with RLHF ([IBM](https://www.ibm.com/think/topics/instruction-tuning), [Toloka](https://toloka.ai/blog/base-llm-vs-instruction-tuned-llm)) | LLaMA‑2‑chat (Meta) |
| **Chat Models**      | Dialogue transcripts, conversation logs | Instruct tuning + multi-turn scenario alignment, system prompts ([ScrapingAnt](https://scrapingant.com/blog/llm-instruct-vs-chat), [Wikipedia](https://en.wikipedia.org/wiki/Claude_%28language_model%29)) | Claude 3 series (Anthropic) |
| **Code / Coder**     | Code repositories + natural language comments | Fine-tuned on NL↔code datasets; often with execution/test feedback ([arXiv](https://arxiv.org/abs/2002.08155), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2949761224001147)) | Code Llama, CodeBERT |
| **Reasoning / Thinking** | Complex question–answer, chain-of-thought examples | Prompting with chain-of-thought or hybrid training plus external verification ([arXiv](https://arxiv.org/abs/2201.11903), [Wired](https://www.wired.com/story/anthropic-world-first-hybrid-reasoning-ai-model)) | Claude 3.7 hybrid reasoning |
| **Multimodal**       | Paired text + image/audio/video datasets | Pretrain on multiple modalities; supervised fine-tuning on aligned tasks ([Wikipedia](https://en.wikipedia.org/wiki/Gemini_%28language_model%29), [BentoML](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)) | GPT‑4, Phi‑4‑multimodal‑instruct ([Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)) |
| **Domain‑specific**  | Specialized data (e.g. medical texts, legal corpora) | Fine-tuning base or instruct model on curated domain dataset | Med‑PaLM (Google), Legal‑LLMs |

## [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)

<a href="images/LoRA.png">
  <img src="images/LoRA.png" width="40%">
</a>

- Instead of updating all model weights (like full fine-tuning), **LoRA freezes the base model** and trains **small adapter matrices**.
- These adapters are **low-rank matrices** added to key weights (e.g., W → W + A×B), requiring **far fewer parameters**.
- LoRA is **faster, cheaper, and uses less memory** than full fine-tuning, and the base model stays reusable.
- 🧠 Use case: Efficiently adapt large models to new tasks/domains with small memory/storage footprint.

### Tiny Numerical Example (LoRA)
- Suppose your model has a weight matrix **W** of shape `512 × 512` → **262,144** parameters.
- With **LoRA**, you freeze `W` and add two small matrices:
  - **A**: `512 × 4`
  - **B**: `4 × 512`
- Trainable parameters: `4 × 512 + 512 × 4 = 4,096` → **64× fewer!**
- The modified weight:  
  **W′ = W + α × (A × B)**  
  where `α` is a scaling factor (e.g. 16).
- ✅ `W` stays untouched; only `A` and `B` are learned.

## Function Calling
Function calling allows the model to:
- Know when and how to use tools (functions/APIs) during a conversation.
- Output structured JSON instead of plain text, making it easier for your code to handle responses.
- Let the LLM decide when to call a function, what function to call, and what arguments to pass.

<a href="images/FunctionCalling.png">
  <img src="images/FunctionCalling.png" width="40%">
</a>

[Function calling - OpenAI API](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)

## Prompt techniques
[https://github.com/BachNgoH/PromptingTechniques](https://github.com/BachNgoH/PromptingTechniques)

# MCP
- [MCP](https://modelcontextprotocol.io/docs/getting-started/intro) is an open protocol that standardizes how applications provide context to large language models (LLMs).
- core problem MCP solves: the M×N integration issue


# RAG

# AI Agent vs Agentic AI

- There is **no universally agreed definition** separating *AI Agent* and *Agentic AI* — the distinction often depends on context or how companies choose to market the concept.
([goole cloude](https://cloud.google.com/discover/what-are-ai-agents?hl=vi#what-is-the-difference-between-ai-agents-ai-assistants-and-bots),
[ibm](https://www.ibm.com/think/topics/agentic-ai), [geeksforgeeks](https://www.geeksforgeeks.org/artificial-intelligence/agentic-ai-vs-ai-agents/), ...)

<a href="images/agentic_workflow.png">
  <img src="images/agentic_workflow.png" width="60%">
</a>

[source: weaviate](https://weaviate.io/blog/what-are-agentic-workflows#what-are-agentic-workflows)

## Basic Workflow of an AI Agent
1. **Receive Goal** – Accepts a target or objective from a user or system.
2. **Perceive Environment** – Collects relevant data from inputs, APIs, or sensors.
3. **Plan** – Breaks the goal into steps and decides the sequence of actions.
4. **Act** – Executes actions using tools, APIs, or other systems.
5. **Observe & Evaluate** – Monitors outcomes and gathers feedback.
6. **Adjust** – Modifies plans or actions to improve results or respond to changes.

## Some Agentic workflow patterns
### Planning

<a href="images/planning_pattern.jpg">
  <img src="images/planning_pattern.jpg" width="60%">
</a>

**Why:** Breaks complex tasks into smaller steps, improving reasoning and reducing errors.  
**When:** Use for multi-step problem-solving where the solution path is unclear.

---

### Tool Use

<a href="images/tool_use_pattern.jpg">
  <img src="images/tool_use_pattern.jpg" width="60%">
</a>

**Why:** Gives the agent real-time, external capabilities beyond its training data.  
**When:** Use when tasks need up-to-date information or external system interaction.

---

### Reflection

<a href="images/reflection_pattern.jpg">
  <img src="images/reflection_pattern.jpg" width="60%">
</a>


**Why:** Enables self-evaluation and iterative improvement without human feedback.  
**When:** Use when tasks benefit from multiple refinement cycles for higher accuracy.




# To-Do List

- [ ] Research different **RAG architectures** and their applications.  
- [ ] Study **methods for evaluating LLM-generated content**.  
- [ ] Explore **AI Agent evaluation** techniques.  
- [ ] Learn about **chatbot systems** and their design principles.  
- [ ] Investigate **Agentic RAG** and its use cases.
- [ ] Adjust content order











