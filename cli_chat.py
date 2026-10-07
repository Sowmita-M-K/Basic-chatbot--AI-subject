import sys
from src.nlp_engine import DepartmentChatbot

def main():
    print("=" * 60)
    print("🤖 Welcome to the AI & DS Department Chatbot (CLI Mode)")
    print("=" * 60)
    print("Ask questions about: 3rd year syllabus, labs, HOD, attendance, projects.")
    print("Type 'exit' or 'quit' to stop.\n")

    bot = DepartmentChatbot()

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nBot: Goodbye and good luck with your 3rd year studies! 👋")
                break

            result = bot.get_response(user_input)
            print(f"\nBot [{result['tag']} | confidence: {result['confidence']}]:")
            print(result["response"])
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()
