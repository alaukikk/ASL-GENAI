
import os
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread

import pyttsx3
from groq import Groq


MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """
You are a helpful communication assistant for an ASL
fingerspelling application.

The user communicates by spelling English words using
recognized ASL alphabet letters.

Important rules:
- Preserve the user's intended meaning.
- Correct spelling and grammar conservatively.
- Do not invent information or add details.
- If the text is ambiguous, explain the ambiguity.
- You can translate into Hindi or Marathi when asked.
- You can answer ordinary conversational questions.
- Use clear, simple, respectful language.
"""


class ASLGenAIAssistant:

    def __init__(self, root, recognized_text=""):
        self.root = root
        self.root.title("ASL GenAI Communication Assistant")
        self.root.geometry("760x650")
        self.root.minsize(600, 500)

        self.client = None
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        self.speech_engine = None

        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not configured.")

            self.client = Groq(api_key=api_key)

        except Exception as error:
            messagebox.showerror("API Setup Error", str(error))

        self.create_interface(recognized_text)

    def create_interface(self, recognized_text):

        title = ttk.Label(
            self.root,
            text="ASL GenAI Communication Assistant",
            font=("Arial", 17, "bold")
        )
        title.pack(pady=12)

        ttk.Label(
            self.root,
            text="Recognized text from your ASL alphabet recognizer:"
        ).pack(anchor="w", padx=15)

        self.input_text = tk.Text(
            self.root,
            height=4,
            font=("Arial", 12),
            wrap="word"
        )
        self.input_text.pack(fill="x", padx=15, pady=6)
        self.input_text.insert("1.0", recognized_text)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=8)

        ttk.Button(
            button_frame,
            text="Correct Text",
            command=self.correct_text
        ).grid(row=0, column=0, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="Translate to Hindi",
            command=lambda: self.translate("Hindi")
        ).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="Translate to Marathi",
            command=lambda: self.translate("Marathi")
        ).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(
            self.root,
            text="AI output:"
        ).pack(anchor="w", padx=15, pady=(8, 0))

        self.output_text = tk.Text(
            self.root,
            height=8,
            font=("Arial", 12),
            wrap="word"
        )
        self.output_text.pack(fill="both", expand=True, padx=15, pady=6)

        ttk.Button(
            self.root,
            text="Speak Output",
            command=self.speak_output
        ).pack(pady=5)

        ttk.Separator(self.root).pack(fill="x", padx=15, pady=10)

        ttk.Label(
            self.root,
            text="Ask the AI assistant a question:"
        ).pack(anchor="w", padx=15)

        self.question_entry = ttk.Entry(
            self.root,
            font=("Arial", 12)
        )
        self.question_entry.pack(fill="x", padx=15, pady=6)
        self.question_entry.bind(
            "<Return>",
            lambda event: self.ask_question()
        )

        ttk.Button(
            self.root,
            text="Send to AI",
            command=self.ask_question
        ).pack(pady=6)

        self.status = ttk.Label(
            self.root,
            text="Ready",
            foreground="green"
        )
        self.status.pack(pady=8)

    def get_input(self):
        return self.input_text.get("1.0", "end").strip()

    def show_output(self, text):
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)

    def call_ai(self, messages):
        if self.client is None:
            raise RuntimeError(
                "Groq API is not configured. Check GROQ_API_KEY."
            )

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=700
        )

        return response.choices[0].message.content.strip()

    def run_task(self, prompt, save_history=False):

        if not self.get_input() and not save_history:
            messagebox.showwarning(
                "Missing text",
                "Please enter some text first."
            )
            return

        self.status.config(text="AI is processing...")
        self.root.update_idletasks()

        def worker():
            try:
                if save_history:
                    self.history.append({
                        "role": "user",
                        "content": prompt
                    })
                    messages = self.history.copy()
                else:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]

                result = self.call_ai(messages)

                if save_history:
                    self.history.append({
                        "role": "assistant",
                        "content": result
                    })

                self.root.after(
                    0,
                    lambda: self.finish_task(result)
                )

            except Exception as error:
                self.root.after(
                    0,
                    lambda: self.finish_error(str(error))
                )

        Thread(target=worker, daemon=True).start()

    def finish_task(self, result):
        self.show_output(result)
        self.status.config(text="Response received", foreground="green")

    def finish_error(self, error):
        self.status.config(text="Request failed", foreground="red")
        messagebox.showerror("AI Error", error)

    def correct_text(self):
        text = self.get_input()
        if not text:
            return

        prompt = (            "You are correcting text produced by an ASL alphabet "
            "fingerspelling recognizer. The input may contain individual "
            "words, incomplete sentences, missing punctuation, or minor "
            "recognition errors.\n\n"
            "Your tasks:\n"
            "1. Correct obvious spelling mistakes.\n"
            "2. Arrange words into natural, grammatically correct English.\n"
            "3. Add suitable capitalization and punctuation.\n"
            "4. Preserve the user's original meaning and tone.\n"
            "5. Never invent facts or add missing information.\n"
            "6. If multiple meanings are possible, ask for clarification "
            "instead of guessing.\n\n"
            "Return the corrected message only when its meaning is clear.\n\n"
            "Recognized text: " + text        )
        self.run_task(prompt)

    def translate(self, language):
        text = self.get_input()
        if not text:
            return

        prompt = (
            f"Translate the following message into {language}. "
            "Preserve its meaning. Return the translation and "
            "nothing else. If the input is unclear, say so.\n\n"
            + text
        )
        self.run_task(prompt)

    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            return

        self.question_entry.delete(0, "end")
        self.run_task(question, save_history=True)

    def speak_output(self):
        text = self.output_text.get("1.0", "end").strip()

        if not text:
            messagebox.showinfo(
                "No text",
                "Generate some AI output before using speech."
            )
            return

        def speak():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as error:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Speech Error", str(error)
                    )
                )

        Thread(target=speak, daemon=True).start()


def open_assistant(recognized_text=""):
    root = tk.Tk()
    ASLGenAIAssistant(root, recognized_text)
    root.mainloop()


if __name__ == "__main__":
    open_assistant()
