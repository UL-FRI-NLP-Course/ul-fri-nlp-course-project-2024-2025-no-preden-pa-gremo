import threading
import itertools
import sys
import time
from datetime import datetime
from Data.readData import get_final_traffic_text  # import the function
from LLMs.gaMS import chat_with_gams

def spinner(text, stop_event):
    for c in itertools.cycle('|/-\\'):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{text} {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(text) + 2) + '\r')  # Clear line

def step_one():
    while True:
        date_input = input("Enter date (YYYY-MM-DD HH:MM:SS): ")
        print() 
        if date_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            return None
        try:
            user_date = datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
            # Spinner setup
            stop_event = threading.Event()
            spinner_thread = threading.Thread(target=spinner, args=("Reading news from excel", stop_event))
            spinner_thread.start()
            # Call get_final_traffic_text with the input date string
            traffic_report = get_final_traffic_text(date_input)
            stop_event.set()
            spinner_thread.join()
            if traffic_report is None:
                print("There are no reports for the inputted time.")
                continue  # ask for date again
            else:
                print(traffic_report)
                response = chat_with_gams(traffic_report)
                print(f"GaMS: {response}")
                return user_date
        except ValueError:
            print("Invalid format. Please enter date as YYYY-MM-DD HH:MM:SS.")

def main():
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("------------- News Generation Tool -------------")
    print("Welcome to the news generation tool! \n"
          "Type 'exit' or 'quit' to stop. \n"
          "Type 'next' to generate a new news article.")
    print("------------------------------------------------")

    while True:
        user_date = step_one()
        if user_date is None:
            break

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                return
            if user_input.lower() == "next":
                break
            print(f"Echo: {user_input}")

if __name__ == "__main__":
    main()