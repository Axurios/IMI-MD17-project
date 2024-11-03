from gui import GUI
from helper import check_requirements

if __name__ == "__main__":
    # Check if packages from "requirements.txt" are installed # noqa
    check_requirements()
    # Create and run the application
    app = GUI()
