import ollama

MODEL = "qwen2.5:1.5b"

roles = {
    "1": {
        "name": "Python Tutor",
        "prompt": "You are a patient Python tutor who explains only python concepts simply."
    },
    "2": {
        "name": "Fitness Coach",
        "prompt": "You are a motivating fitness coach giving practical advice only related to fitness."
    },
    "3": {
        "name": "Travel Guide",
        "prompt": "You are a knowledgeable travel guide suggesting places and tips."
    }
}


def choose_role():
    print("\nAvailable Roles:")
    for key, role in roles.items():
        print(f"{key}. {role['name']}")

    choice = input("Pick a role (number): ")
    return roles.get(choice)


def main():
    current_role = choose_role()

    if not current_role:
        print("Invalid choice. Exiting.")
        return

    print(f"\nRole set: {current_role['name']}")
    print("Type your message (or 'switch' to change role, 'quit' to exit)\n")


    messages = [
        {"role": "system", "content": current_role["prompt"]}
    ]

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "switch":
            current_role = choose_role()
            if not current_role:
                print("Invalid choice. Continuing with previous role.")
                continue

            print(f"\nRole set: {current_role['name']}")
            
            messages = [
                {"role": "system", "content": current_role["prompt"]}
            ]
            continue

        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(
            model=MODEL,
            messages=messages
        )

        reply = response["message"]["content"]

        print(f"{current_role['name']}: {reply}\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()