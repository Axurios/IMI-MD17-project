"""
This module is the main module of the program,
and can be executed to run the needed part of the project
"""
from tkinter import Tk, Label, Button, StringVar
from helper import check_requirements, select_file
# import Descriptor
selected_file_path = None  # Variable to store the selected file path


if __name__ == "__main__":
    check_requirements()
    # Initialize main window
    fenetre = Tk()
    fenetre.title("ATK title")

    # Label to display the selected file name
    file_name_var = StringVar()
    file_name_var.set("No file selected")  # Default text
    file_name_label = Label(fenetre, textvariable=file_name_var)
    file_name_label.pack()

    # Button to trigger file selection, using lambda to capture file_name_var
    button = Button(fenetre, text="Select CSV File", command=lambda: select_file(file_name_var)) # noqa:
    button.pack()

    # Main loop
    fenetre.mainloop()

    # After the window closes, get last selected file path if available
    selected_file_path = select_file(file_name_var)
    if selected_file_path:
        print(f"Selected file path: {selected_file_path}")
        # Further processing of the file can be done here
    else:
        print("No file selected.")
