# Resume RAG — Gemini Research Assistant

Welcome to **Resume RAG**, a smart assistant designed to help you analyze, summarize, and compare resumes with the power of **Gemini 1.5**. This project showcases **retrieval-augmented generation (RAG)** applied to resumes, enabling you to extract insights — all from uploaded PDFs or TXT files.

---

## 🌟 Features

- **Multi-step reasoning**: Uses `tree_summarize` to provide thorough insights from resumes.  
- **Gemini embeddings**: Leverages `embedding-001` to understand candidate skills, experience, and education.  
- **Template queries**: Predefined questions for instant analysis:
  - Summarize key skills and experience.  
  - Highlight education and certifications.  
  - Compare multiple candidates.  
  - Identify potential gaps or weaknesses.  
- **Interactive chat**: Ask your own queries or use template questions seamlessly.  
- **Full source trace**: See which part of the resume contributed to each insight.  
- **Clear & download chat**: Keep your workspace clean or export your analysis.

---

## 📂 How It Works

1. **Upload Resumes**: Upload one or multiple PDF/TXT resumes via the sidebar.  
2. **Build Index**: Each resume is processed and stored in a vector index for semantic search.  
3. **Query**: Ask questions directly or click on template queries — the assistant analyzes the resumes and provides concise insights.  
4. **Trace & Sources**: Optional reasoning trace shows which parts of the resumes contributed to the answer.  
5. **Clear or Download**: Reset the chat or download the conversation for record-keeping.
