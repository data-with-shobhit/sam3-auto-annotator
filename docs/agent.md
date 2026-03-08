# AI Agent Developer Guide (`agent.md`)

This repository was designed specifically to integrate seamlessly with AI coding assistants (like Claude, Gemini, and Antigravity). 

If you are an AI agent operating on this codebase, **adhere strictly to the architectural constraints outlined in this document.**

---

## 🛑 Strict Architectural Constraints

### 1. No UI Blocking Components
This is a Streamlit application. **Streamlit reruns the entire Python script top-to-bottom on every interaction.**
- DO NOT import heavy ML libraries (`torch`, `ultralytics`, `sam3`) at the top of `app.py` or any `ui/pages/` files.
- You **MUST** use delayed/local imports (e.g., `import torch` *inside* the function definition or *inside* the `if st.button()` block). If you globally import them, the UI will freeze for 3 seconds on every button click.

### 2. CUDA Multiprocessing Safety
- When invoking parallel GPU tasks, **NEVER** use Python's default `threading` or `multiprocessing` without explicitly setting the context to `spawn`. 
- PyTorch CUDA contexts cannot be safely forked. If a parent process (Streamlit) initializes a CUDA context, all subprocesses will crash with `Cannot re-initialize CUDA in forked subprocess`.
- Ensure all heavy ML code executes in completely isolated scripts (like `scripts/annotate_worker.py`) executed via `subprocess.Popen` or `ProcessPoolExecutor(mp_context="spawn")`.

### 3. State Management
- Do not use global variables.
- All persistent data (uploaded file paths, project names, model architectures) MUST be stored in `st.session_state`. 
- If adding a new page, verify that the state variables you need exist by using `if 'var' not in st.session_state:` fallback logic.

### 4. Logging & Formatting
- **Do not use `print()` statements in production code.**
- All logs must be routed through the centralized logger defined in `config/logging_config.py`. 
- Import: `from config.logging_config import setup_logging`
- Instantiate: `log = setup_logging("component_name")`
- Render: `log.info("Status update")`

### 5. Handling File Paths
- Use `pathlib.Path` exclusively instead of `os.path`.
- Always reference `BASE_DIR`, `PROJECTS_DIR`, and `LOGS_DIR` from `config/settings.py`. Do not hardcode raw absolute paths.

---

## 🛠️ Adding New Features

### How to Add a New Page
1. Create a logic file in `ui/pages/new_page_name.py`.
2. Define a main rendering function, e.g., `def render_new_page():`.
3. Open `ui/components/sidebar.py` and register your page in the dictionary.
4. Open `app.py`, intercept the new state key in the routing loop, and lazy import your `render_new_page` function.

### How to Add a New Background Task
1. Write pure Python, CLI-executable code in a new file under the `scripts/` directory (so it has its own isolated memory space).
2. The script should accept arguments via `argparse` or read from a `requests.json` dump file path passed as an argument.
3. Use `subprocess.Popen()` in your Streamlit page to invoke the script. Track its PID to ensure it doesn't become a zombie process.
4. Establish communication using lightweight file polling (e.g., watching a `.csv` or `.json` file for updates) rather than complex socket communication.
