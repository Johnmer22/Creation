import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
data = pd.read_csv("dataset.csv")
responses = data["response"].tolist()

# Encode responses using SBERT
response_embeddings = model.encode(responses)

def get_best_response(user_query, threshold=0.5):
    """
    Finds the best response for a given user query based on cosine similarity.
    """
    query_embedding = model.encode([user_query], batch_size=1)[0]
    similarities = cosine_similarity([query_embedding], response_embeddings)[0]
    max_similarity = max(similarities)
    if max_similarity < threshold:
        return "I'm not sure about that. Can you rephrase?"
    return responses[similarities.argmax()]

def handle_query():
    """
    Handle user query, generate response, and update the chat log.
    """
    user_query = user_input.get()
    if user_query.strip():
        # Display user query in blue
        chat_log.insert(tk.END, f"You: {user_query}\n", "user")
        bot_response = get_best_response(user_query)
        # Display bot response in green
        chat_log.insert(tk.END, f"Bot: {bot_response}\n\n", "bot")
        user_input.delete(0, tk.END)
        chat_log.see(tk.END)  # Automatically scroll to the latest message

# Create the main GUI window
window = tk.Tk()
window.title("Chatbot")

# Resize the main window
window.geometry("700x500")  # Width x Height in pixels

# Configure grid layout to be resizable
window.grid_rowconfigure(0, weight=1)  # Make the chat log row stretchable
window.grid_columnconfigure(0, weight=1)  # Make the chat log column stretchable

# Chat log (scrollable text box)
chat_log = scrolledtext.ScrolledText(window, wrap=tk.WORD, state="normal", font=("Arial", 12))
chat_log.grid(column=0, row=0, columnspan=2, padx=10, pady=10, sticky="nsew")  # Stretch in all directions

# Define text tags for coloring
chat_log.tag_configure("user", foreground="blue")  # User messages in blue
chat_log.tag_configure("bot", foreground="green")  # Bot messages in green

# Initial welcome message from the chatbot
chat_log.insert(tk.END, "Bot: Hello! Welcome to the chatbot. How can I assist you today?\n\n", "bot")
chat_log.see(tk.END)

# User input box
user_input = tk.Entry(window, font=("Arial", 12))
user_input.grid(column=0, row=1, padx=10, pady=10, sticky="ew")  # Stretch horizontally

# Send button
send_button = tk.Button(window, text="Send", font=("Arial", 12), command=handle_query)
send_button.grid(column=1, row=1, padx=10, pady=10, sticky="e")

# Adjust column widths to maintain proper proportions
window.grid_columnconfigure(0, weight=3)  # Make the input field take more space
window.grid_columnconfigure(1, weight=1)  # Make the button smaller

# Bind the Enter key to the user input box
user_input.bind("<Return>", lambda event: handle_query())

# Start the GUI event loop
window.mainloop()






